# Bug: Yoke LR schedulers restart their warmup on continuation

Found 2026-11-13 by `yoke-run-doctor` while checking the TPBF smoke run
(`dev_plans/tpbf_meltpool_training.md`, Run card doctor log). Status: **open**.
It is pre-existing and not specific to CPU mode or the TPBF harness. The fix
is deferred by the user.

## Summary

All three schedulers in `src/yoke/lr_schedulers.py`
(`WarmUpPowerDecayScheduler`, `CosineWithWarmupScheduler`,
`ConstantWithWarmupScheduler`) compute the LR from `self._step_count`
(`src/yoke/lr_schedulers.py:145`, `:200`, `:241`). `_step_count` is PyTorch
bookkeeping that `torch.optim.lr_scheduler.LRScheduler.__init__` **always
resets to 1**, whatever `last_epoch` is passed. So when a continuation job
rebuilds the scheduler with `last_epoch = compute_last_epoch(...)`, the
schedule starts again from step 1. The LR collapses to the warmup start and
warms up again, instead of continuing where the previous job stopped.

A second, smaller defect is in `yoke.utils.builders.compute_last_epoch`
(`src/yoke/utils/builders.py:91`). It returns
`steps_per_epoch * (starting_epoch - 1)`, but after `starting_epoch` full
epochs the scheduler has taken `steps_per_epoch * starting_epoch` steps. With
the constructor's own `+1` step that is one epoch short (observed: 9482 vs
9500 after 500 epochs of 19 steps). Today it is masked by the bigger bug.

## Impact

- Every harness that uses one of these schedulers **and** spans more than one
  cycle (`total_epochs > cycle_epochs`) gets a sawtooth LR: a fresh warmup,
  then the early part of the cosine, in every job. A cosine run never
  reaches its low-LR tail. The loss curves may show a bump or plateau at
  each cycle boundary.
- Users: `HarnessTrainer` harnesses that pass a `scheduler_builder` with these
  classes, plus older bespoke scripts. Callers today (grep, 2026-11-13): 21
  harness scripts under `applications/harnesses/` (`ch_*`, `se_*`, `vt_*`,
  `lsc_action`, `ro_tpbf_mlp`), and the Lightning module in
  `src/yoke/models/vit/swin/bomberman.py` (`lr_scheduler` default) and
  `src/yoke/models/vit/swin/diffusion_bomberman.py`. Check the Lightning path
  separately: Lightning restores scheduler `state_dict`s itself, which may
  restore `_step_count`, and may not be affected.
- Past GPU studies with several cycles probably trained with this sawtooth.
  Their results are still valid runs, but not the intended schedule.
- Current workaround: run one cycle per study (`cycle_epochs >=
  total_epochs`). The TPBF production run does this (`CYCLE_EPOCHS=1000`).

## How to find it (for an agent)

1. Read `src/yoke/lr_schedulers.py`. Each `get_lr` calls its `*_lr_calc` with
   `step=self._step_count`.
2. Read the PyTorch base class (installed torch: `python -c "import inspect,
   torch.optim.lr_scheduler as s; print(inspect.getsource(s.LRScheduler.__init__))"`).
   `__init__` sets `self._step_count = 0`, then `_initial_step()` calls
   `step()` once, which makes it 1. `last_epoch` is independent of it.
3. Read `HarnessTrainer.setup` (`src/yoke/harnesses/trainer.py`, the
   `# --- LR scheduler ---` block). On continuation it builds a **new**
   scheduler with `last_epoch=compute_last_epoch(starting_epoch,
   steps_per_epoch)`. It never loads a scheduler `state_dict`; checkpoints
   don't store one (`yoke.utils.checkpointing.save_model_and_optimizer`).
4. The existing tests (`tests/test_LRschedulers.py`) only step fresh
   schedulers, so they don't catch it.

## How to verify it (reproduction, CPU, under a second)

Run from any directory (not inside `runs/`) with the Yoke environment:

```python
import torch
from yoke.lr_schedulers import CosineWithWarmupScheduler as C
from yoke.utils.builders import compute_last_epoch

kw = dict(anchor_lr=1e-3, terminal_steps=19000, warmup_steps=200,
          num_cycles=0.5, min_fraction=0.01)
m = torch.nn.Linear(2, 2)
opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
s = C(opt, **kw)
for _ in range(9500):            # cycle 1: 500 epochs x 19 steps
    opt.step(); s.step()
print("end of cycle 1:", s.last_epoch, s._step_count, opt.param_groups[0]["lr"])

# Continuation, exactly as HarnessTrainer does it: reload the optimizer, then
# build a new scheduler with last_epoch from compute_last_epoch.
opt2 = torch.optim.AdamW(m.parameters(), lr=1e-3)
opt2.load_state_dict(opt.state_dict())
s2 = C(opt2, last_epoch=compute_last_epoch(500, 19), **kw)
print("start of cycle 2:", s2.last_epoch, s2._step_count, opt2.param_groups[0]["lr"])
```

Observed on 2026-11-13 (torch 2.15.0):

```
end of cycle 1:   last_epoch 9500  _step_count 9501  lr 0.000513...
start of cycle 2: last_epoch 9482  _step_count 1     lr 5e-06
```

**Expected after a fix:** the cycle-2 start LR equals the next value of the
uninterrupted schedule (about 5.13e-4), and `last_epoch` is 9500.

Also note: building a scheduler with `last_epoch >= 0` on an optimizer that
was **not** loaded from a checkpoint raises `KeyError: "param 'initial_lr' is
not specified ..."` (PyTorch needs `initial_lr` from the saved optimizer
state). The fix and the tests must keep using a reloaded optimizer.

End-to-end confirmation in a real harness: a study with `TOTAL_EPOCHS = 2 x
CYCLE_EPOCHS`. Add `print(optimizer.param_groups[0]["lr"])` in a test-only
`on_epoch_start` hook, or log the LR. The first epoch of cycle 2 shows the
warmup LR.

## Fix plan

Goal: a continued run follows the same LR trajectory as an uninterrupted
run, for all three schedulers, with no change to fresh runs.

1. **Schedulers** (`src/yoke/lr_schedulers.py`): compute the step from
   `self.last_epoch`, not `self._step_count`. With PyTorch's convention,
   `last_epoch` is the number of `step()` calls so far. After construction
   it is `last_epoch_arg + 1` (0 for a fresh run), and the current
   code's `_step_count` is `last_epoch + 1` on a fresh run. So use
   `step = self.last_epoch + 1` to keep fresh-run values **bit-identical**
   to today's. Apply this in `get_lr` of all three classes. Add a comment
   explaining why `_step_count` must not be used.
2. **`compute_last_epoch`** (`src/yoke/utils/builders.py`): after `starting_epoch`
   completed epochs, `steps_per_epoch * starting_epoch` steps were taken.
   PyTorch's constructor adds one step when it builds a scheduler with a
   given `last_epoch`, so the argument must be
   `steps_per_epoch * starting_epoch - 1`. Fresh runs stay `-1`. Update its
   docstring and `tests/utils/test_builders.py` (the current expected values
   `0` and `500` encode the off-by-one).
   - Check the meaning of `starting_epoch` first: `HarnessTrainer.train`
     does `starting_epoch += 1` *after* setup, and the checkpoint stores the
     last completed epoch. So at setup time `starting_epoch` = completed
     epochs. Confirm this in `build_from_checkpoint` /
     `load_model_and_optimizer`.
3. **Optional, more robust:** save the scheduler `state_dict` in the
   checkpoint (`extra_state={"scheduler_state_dict": ...}`) and load it on
   continuation. Restoring it would fix both problems directly, but older
   checkpoints lack it, so a fallback to steps 1-2 is still needed. Decide
   whether it's worth it after steps 1-2.
4. **Bespoke scripts:** grep for scripts that build schedulers themselves
   (`grep -rn "Scheduler(" applications/harnesses`). Check each computes
   `last_epoch` like `compute_last_epoch`, or switch it to that helper.
5. **Lightning:** check whether `ch_lightning_loderunner` (bomberman
   Lightning module) restores scheduler state from its checkpoints. If so,
   steps 1-2 must not double-count there.

## Tests to add

- `tests/test_LRschedulers.py`: for each of the three schedulers, step a
  fresh scheduler N times, recording the LR. Then rebuild at step k (reloaded
  optimizer, `last_epoch=k-1`) and step on. The LR sequence must equal the
  uninterrupted one (`pytest.approx`). Use small N (for example 50), a short
  warmup, and k inside both the warmup and the decay.
- A regression test that fresh-run LR sequences are unchanged (compare
  with `cos_decay_lr_calc` etc. called at steps 1..N).
- `tests/utils/test_builders.py`: updated `compute_last_epoch` values.
- `tests/harnesses/test_trainer_preflight.py`: extend the CPU-mode
  continuation test (`test_cpu_mode_continuation`) to use
  `CosineWithWarmupScheduler`. Assert that the LR at the start of cycle 2
  equals the uninterrupted schedule's LR at that step.

## Verification

```bash
./.venv/bin/python -m pytest -Werror tests/test_LRschedulers.py \
    tests/utils/test_builders.py tests/harnesses -q
./.venv/bin/python -m pytest -Werror -q          # full suite
./.venv/bin/ruff check src/yoke/lr_schedulers.py src/yoke/utils/builders.py
./.venv/bin/ruff check --preview src/yoke/lr_schedulers.py src/yoke/utils/builders.py
./.venv/bin/ruff format --check src/yoke/lr_schedulers.py src/yoke/utils/builders.py
```

Rerun the reproduction script above: the cycle-2 start LR must be about
5.13e-4. Then a real check: the TPBF smoke CSV
(`applications/harnesses/ro_tpbf_mlp/tpbf_kfold_smoke.csv`, with new study
IDs) with an LR log line, on rocinante `debug`.

## After the fix

- Add the issue (and its FIXED date) to
  `dev_plans/known_issues_from_skills_work.md`.
- Add a failure signature to
  `skills/yoke-run-doctor/reference/failure_signatures.md`: "loss jumps or
  plateaus at every cycle boundary; LR restarts at warmup".
- The TPBF harness can then go back to several cycles per study if wanted.
