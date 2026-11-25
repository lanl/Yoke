# HarnessTrainer CPU single-process mode (for CPU-only clusters)

Started 2026-11-13 during the TPBF melt-pool work
(`dev_plans/tpbf_meltpool_training.md`). Approved direction: option A, a CPU
mode inside `HarnessTrainer`, plus skill and doc updates so agents know about
it.

## Goal

Let a `HarnessTrainer` harness run as **one process on one CPU node**, with
no GPU, no NCCL, and no `srun`-launched ranks, for CPU-only machines such as
rocinante and for small models. Keep everything else the trainer gives:
`.pth` checkpoints, continuation and resubmission, record files, and
`--preflight`. GPU harnesses must behave exactly as before.

## Why the trainer fails on CPU today (verified by reading the source)

- `HarnessTrainer.run()` always calls `setup_distributed()`. That reads the
  SLURM rank variables, calls `torch.cuda.set_device`, and initializes the
  GPU-only `nccl` backend (`src/yoke/utils/parallel.py:17-62`).
- Epoch timing calls `torch.cuda.synchronize(self.device)`
  (`src/yoke/harnesses/trainer.py:405`, `:414`).
- Everything else is already CPU-safe. Preflight uses
  `setup_single_process_cpu()` (`gloo`, world size 1,
  `src/yoke/utils/parallel.py:65`). The DDP wrap has a CPU branch
  (`trainer.py:307`). In a one-process group, the `all_gather` in the
  datasteps and the checkpoint barriers are no-ops. `pin_memory` is
  already conditional on `torch.cuda.is_available()`.

## Design

`src/yoke/harnesses/trainer.py`:

1. New keyword argument `device_type: str = "cuda"`, validated against
   `("cuda", "cpu")` (a `ValueError` otherwise, matching the existing
   `evaluate_checkpoint` check). Documented in the class docstring.
2. `setup_distributed()`: when `device_type == "cpu"`:
   - if `int(os.environ.get("SLURM_NTASKS", "1")) > 1`, raise
     `RuntimeError("CPU mode runs one process; launch with --ntasks=1 ...")`.
     This stops N duplicate copies, all writing the same records,
     checkpoints and resubmissions, when a template uses `srun` with
     several tasks.
   - otherwise call `setup_single_process_cpu()`, which gives `gloo`, rank 0,
     world size 1, and the CPU device.
   - the `cuda` path is unchanged.
3. `train()`: call `torch.cuda.synchronize` only when
   `self.device.type == "cuda"`. `dist.barrier()` stays; it is cheap with
   `gloo` and one rank.
4. No change to `setup()`, `finalize()`, `preflight()`, the datasteps, or
   the checkpointing.

CPU threads are left to the job environment (`OMP_NUM_THREADS`, which torch
reads at start-up), set in the SLURM template. A 17k-parameter MLP is
probably fastest with few threads. The smoke run will show this.

## Tests (`tests/harnesses/`)

In `tests/harnesses/test_trainer_preflight.py`, which already uses a real
`gloo` group and `train_DDP_array_epoch`:

- `device_type="cpu"` full `run()` on the tiny array dataset, with
  `resubmit=False` and `total_epochs=2`, `cycle_epochs=2`, in `tmp_path`.
  Check that records are written, the `.pth` checkpoint exists and reloads,
  and the process group is torn down. Monkeypatch `torch.cuda.synchronize`
  and `torch.cuda.set_device` to raise, which proves no CUDA call is made.
- Continuation in CPU mode: a second `run()` with `continuation=True` from
  that checkpoint resumes at epoch 3.

In `tests/harnesses/test_trainer.py`:

- an invalid `device_type` raises `ValueError`;
- CPU mode with `SLURM_NTASKS=4` raises `RuntimeError` (monkeypatched env);
- the default `device_type` is still `"cuda"` and still calls the patched
  `setup_distributed` (an existing-behaviour guard).

## Docs and skills to update

| File | Change |
| --- | --- |
| `AGENTS.md` §5a | "always DDP" becomes "DDP on GPUs, or `device_type="cpu"` for one process on a CPU node". Short paragraph on when to use it. |
| `docs/source/harness_trainer.rst` | Same wording at "always DDP". New short section, "CPU-only machines", with the constructor call and the one-task SLURM rule. |
| `skills/yoke-train-my-data/SKILL.md` | Round 1: ask whether the cluster has GPUs or is CPU-only. Step 2: say CPU-only means CPU mode. |
| `skills/yoke-model/SKILL.md` | Compute question: GPU memory, *or* CPU node cores/memory. Note that CPU mode suits small models. |
| `skills/yoke-new-harness/SKILL.md` | Round 4 (submission): ask GPU vs CPU-only; CPU gives `device_type="cpu"`, one task per job, no `NGPUS`/`KNODES` columns needed. Verify step: "multi-GPU behaviour" applies only to GPU mode. |
| `skills/yoke-new-harness/reference/harness_checklist.md` | Add `device_type=` to the `HarnessTrainer(...)` example, with a one-line rule. |
| `skills/yoke-new-harness/reference/study_csv_and_templates.md` | `KNODES`/`NGPUS` are conventional for GPU templates only. List the CPU template's required tokens. |
| `skills/yoke-slurm-template/SKILL.md` | Step 1/4: if the chosen partition has no GPU GRES, use the CPU variant. Skip the GPU, NCCL and binding questions; ask threads instead. The step 6 token check differs by variant. |
| `skills/yoke-slurm-template/reference/slurm_template_skeleton.md` | Add a "CPU-only training template" section: `--nodes=1 --ntasks=1 --cpus-per-task=@@...@@`, no `--gpus-per-node`, no `MASTER_ADDR`/NCCL/`nvidia-smi`, `OMP_NUM_THREADS`, and `python -u` (no multi-rank `srun`). Required tokens: `<studyIDX>`, `<epochIDX>`, `<train_script>`, `<INPUTFILE>`. |
| `skills/yoke-slurm-template/reference/slurm_queries.md` | §3: a `Gres` of `(null)` means a CPU-only partition. The derived defaults then come from cores, not GPUs. §6 (NCCL) is not needed for CPU mode. |
| `skills/yoke-slurm-template/reference/machine_profile_template.md` | Partition table "GPUs/node" may be `none`; add a "CPU-only: use HarnessTrainer device_type=cpu" quirk line. |
| `skills/shared/cards.md` | Slurm card: new `device:` field (`cuda` / `cpu`). Run card `budget:` becomes "GPU hours, or node hours for CPU mode". Model card memory note: "per GPU, or per CPU node". |
| `skills/yoke-launch-study/SKILL.md` + `reference/launch_checks.md` | The budget becomes GPU hours *or* node hours (CPU: `nodes x wall x cycles`). The not-checked list mentions NCCL only for GPU mode. |
| `skills/yoke-run-doctor/reference/failure_signatures.md` | Add: a GPU-mode harness on a CPU node (`Torch not compiled with CUDA enabled` / `No CUDA GPUs are available` / NCCL init errors) means set `device_type="cpu"`; and the new CPU-mode `RuntimeError` about `--ntasks=1`. |

`tests/skills/test_skill_files.py` must still pass. It checks that the repo
paths and `yoke.` names mentioned exist and import, so any new mention
(for example `yoke.utils.parallel.setup_single_process_cpu`) must be
accurate.

## Steps (in order)

1. Trainer change and tests. Run `pytest -Werror tests/harnesses`.
2. `AGENTS.md`, then `docs/source/harness_trainer.rst`.
3. Skill files, in the table order.
4. Full check (below).

## Verification

```bash
./.venv/bin/python -m pytest -Werror tests/harnesses tests/skills -q
./.venv/bin/python -m pytest -Werror -q            # full suite, regressions
./.venv/bin/ruff check src/yoke/harnesses tests/harnesses
./.venv/bin/ruff check --preview src/yoke/harnesses tests/harnesses
./.venv/bin/ruff format --check --diff src/yoke/harnesses tests/harnesses
```

Not verified here: a real rocinante job. The smoke study in the TPBF
harness is the first real test of CPU mode (including one continuation).

## Out of scope (noted, not done)

- Multi-process CPU DDP (`gloo` across tasks or nodes).
- Separate train and validation batch sizes in `HarnessTrainer`. This is
  relevant to the TPBF holdout folds of 19-20 rows with `drop_last=True`. It
  is decided in the harness design, not here.

## Results (2026-11-13)

Done as planned. Files changed: `src/yoke/harnesses/trainer.py`,
`tests/harnesses/test_trainer.py` (3 tests), and
`tests/harnesses/test_trainer_preflight.py` (2 real CPU `run()` tests: a full
run, and a continuation with CUDA calls patched to fail). Also updated:
`AGENTS.md` §5a, `docs/source/harness_trainer.rst` (new "CPU-only machines"
section), and the skill files listed in the table. The new
"CPU-only training template" is in `slurm_template_skeleton.md`, and two new
failure signatures are in `failure_signatures.md`.

- `pytest -Werror tests/harnesses tests/skills`: 166 passed, 1 skipped.
- Full `pytest -Werror`: 886 passed, 1 skipped.
- ruff check / --preview / format --check on the changed src and tests: clean.
- Test detail: `train_DDP_array_epoch` calls `LRsched.step()` every batch,
  so it needs a scheduler; `HarnessTrainer`'s default `None` would crash
  there. The tests pass a constant `LambdaLR`. This is a pre-existing quirk,
  not CPU-specific. A harness using the array epoch must supply a
  `scheduler_builder`.
- Not verified: a real rocinante job.
