# Agent Skills, Phase 2: From "I Have Data" to a Submitted Run

## Revision history

- **Rev 1** (2026-11-10): initial draft. Follows
  `dev_plans/agent_skills_dataset_and_harness.md` (phase 1, implemented).
- **Rev 2** (2026-11-10), after user review:
  - CPU preflight on login nodes is assumed acceptable for now; revisit
    after HPC testing.
  - `model_catalog.md` moves into `yoke-model` (approved).
  - The fix for known issue 7 (loss not moved to the device) is in scope.
  - A user-facing docs page, "Using Yoke with an AI agent", is in scope.
  - The config-driven-harness idea is written up separately in
    `dev_plans/config_driven_harness_idea.md`, for after this phase.
- **Rev 3** (2026-11-11), after user review: `yoke-launch-study` may submit
  the study for the user, if the user says so at launch time. See "Rev 3:
  optional agent submission" below. Implemented 2026-11-11.

## Goal

A user who does **not** know Yoke's internals sits down with Yoke, an AI
coding agent, and their dataset, and leaves with a training run they can
submit themselves, plus a way to come back later and debug it.

Phase 1 covers the middle of that path: dataset, training step, SLURM
template, and harness. Phase 2 adds the beginning, the missing pieces in the
middle, and the end:

| New skill | Covers |
| --- | --- |
| `yoke-train-my-data` | The front door. Frames the problem, routes to the other skills, keeps one progress file |
| `yoke-model` | Choose an existing Yoke model, or compose a new `nn.Module` from Yoke building blocks |
| `yoke-new-loss` | Choose an existing loss, or write a new one, with the right output shape |
| `yoke-launch-study` | On-cluster checks, the one-sample preflight, and the **exact command for the user to run** |
| `yoke-run-doctor` | Analyze and debug a run the user already submitted, in a later session |

Plus two code changes and one docs page:

- a **one-sample preflight mode** for `HarnessTrainer`, which the launch
  skill uses;
- the **fix for known issue 7**: `HarnessTrainer` moves the loss to the
  training device;
- a user-facing page, **"Using Yoke with an AI agent"**
  (`docs/source/agent_skills.rst`), linked from `README.md` and the docs
  index.

## Design principles (decided with the user)

- **The user submits; the agent never does.** *(Superseded by Rev 3: the
  launch skill may submit SLURM studies when the user explicitly asks. See
  "Rev 3: optional agent submission".)* Yoke runs are long and restart
  themselves (`HarnessStudy.continuation_setup` resubmits from the compute
  node). If the agent submitted the job, the run would look tied to the
  agent's terminal session, but it isn't, and the user would lose track of
  it. `yoke-launch-study` stops at printing the command. The user runs it in
  their own shell.
- **Sessions are decoupled through files, not memory.** Everything a later
  session needs is written down: the progress file, the cards, and a **Run
  card** that records the study directories and the command. `yoke-run-doctor`
  starts from those files (or from a study directory the user points at). It
  needs nothing from the session that set the run up.
- **A one-sample forward check is allowed**: build the real dataset from
  sample files, build the model, run **one** forward pass and loss
  evaluation, with no optimizer step and no training. Approved by the user.
- **The phase 1 rules still apply**: interview in rounds, record decisions
  in `dev_plans/`, approval before edits, read-only inspection, no new
  dependencies without approval.

## Flow

```
                      yoke-train-my-data  (progress file: dev_plans/<project>_training.md)
                               │
    ┌──────────────┬───────────┼─────────────┬──────────────────┐
    ▼              ▼           ▼             ▼                  ▼
yoke-new-dataset  yoke-model  yoke-new-loss  yoke-new-training-step (if needed)
    └──────────────┴───────────┴─────────────┴──────────────────┘
                               ▼
                       yoke-new-harness  ──► yoke-slurm-template (if needed)
                               ▼
                       yoke-launch-study  ──► prints command; USER submits
                               ⋮  (hours or days; any later session)
                       yoke-run-doctor   ◄── user: "study 3 died, why?"
```

## Background (verified in the repo unless marked)

- `HarnessTrainer` always runs DDP through
  `yoke.utils.parallel.setup_distributed`. That function reads
  `SLURM_PROCID`/`SLURM_NTASKS`/`SLURM_LOCALID`, `MASTER_ADDR`/`MASTER_PORT`,
  calls `torch.cuda.set_device`, and uses the `nccl` backend
  (`src/yoke/utils/parallel.py:34-57`). So a harness can't currently be run,
  even for one batch, outside a SLURM GPU job.
- Every DDP epoch function checks `if trainbatch_ID >= num_train_batches:
  break` at the top of the training loop (for example
  `epoch/loderunner.py:479`, `epoch/array_output.py:150`), and runs validation
  when `epochIDX % train_per_val == 0`. So `num_train_batches=0`,
  `num_val_batches=1`, and `epochIDX=0` drive exactly **one eval datastep**
  through the harness's real epoch function and `epoch_kwargs`. That is a
  forward pass and a loss, with no backward and no optimizer step.
  `LRsched.step()` runs only inside the training loop, so it is not called.
- `LossBuilder = Callable[[], nn.Module]` (`trainer.py:76`) takes no
  arguments, so a parameterized loss needs a closure or `functools.partial`
  in the harness. The trainer never moves the loss to the device
  (`trainer.py:294`), so a loss with buffers (for example
  `MaskedLossMultiplicative`'s `mask`) would stay on the CPU while the
  predictions are on the GPU. *Inferred from reading, not run.* It is
  recorded as issue 7 in `dev_plans/known_issues_from_skills_work.md` and
  fixed in this phase (see "Code change: move the loss to the device").
- Existing losses (`src/yoke/losses/`): `LpLoss` (`reduction` `none` →
  `(B, C)`, `channel_mean` → `(B,)`, `mean`, `sum`), `NormalizedMSELoss`
  (elementwise with `none`; it normalizes with **batch** statistics over dims
  `(0, 2, 3)`), and the wrappers `MaskedLossMultiplicative` and
  `CroppedLoss2D`, which keep the wrapped loss's shape.
- Model building blocks exist but are undocumented for users:
  `PlainViTBackbone`, `SwinUnetBackbone`, the Swin encoder and transformer,
  patch embedding, `PatchMerge`/`PatchExpand`/`Unpatchify`, `VarEmbed`,
  `PosEmbed`, `TimeEmbed`, `AggVars`, RoPE, `generalMLP`,
  `CNN_Interpretability_Module`/`CNN_Reduction_Module`, and the shape
  helpers `conv2d_shape`/`convtranspose2d_shape`.
- Checkpoints are named `study{NNN}_modelState_epoch{EEEE}.pth`
  (`yoke.utils.builders.checkpoint_name`). Record CSVs and `.out`/`.err` files
  are written in `runs/study_NNN/`, with restart files
  `studyNNN_restart_training_epochMMMM.*` (seen in
  `applications/harnesses/ch_DDP_loderunner/runs/`).
- `yoke-plot-loss` (`src/yoke/cli/plot_loss.py`) plots the record CSVs.

## Code change: preflight mode for `HarnessTrainer`

### What

A `--preflight` flag. The user (or the agent, on sample data) runs the real
train script on a CPU, with the real rendered input file:

```bash
python train_<name>.py @study001_START.input --preflight
```

It builds everything as a real run would, pushes one batch through one eval
datastep, prints a report, and exits. It does no backward pass, takes no
optimizer step, writes no checkpoint, and resubmits nothing.

### How

- `yoke.helpers.cli.add_training_args` gains `--preflight` (`store_true`), so
  every existing harness gets it with no script changes.
- `HarnessTrainer.run()`: if `args.preflight` is set, call a new
  `preflight()` instead of the normal lifecycle:
  1. **Process group**: a single-process `gloo` group on CPU (`world_size=1`,
     `rank=0`, a free local port), instead of the SLURM/NCCL setup. A small
     helper, `yoke.utils.parallel.setup_single_process_cpu()`, sits next to
     `setup_distributed`.
  2. `setup()` as normal, with `device=cpu`. The DDP wrap must use
     `device_ids=None` on CPU, so `setup()` passes `device_ids` only for CUDA
     devices.
  3. Override `batch_size` to `min(args.batch_size, 2)` and `num_workers` to
     0 (fast, and no worker processes).
  4. Call `epoch_fn(**self._epoch_call_kwargs(0))` with
     `num_train_batches=0`, `num_val_batches=1`, `train_per_val=1`, and the
     record filenames pointing at a temporary directory.
  5. Report: model class and parameter count; the batch tuple's shapes and
     dtypes (from one `next(iter(val_dataloader))`); that the loss record
     was written, with its value finite; elapsed time. Exit non-zero, with a
     clear message, on any exception or a non-finite loss.
  6. Tear down the process group.
- `HarnessTrainer` docs (`docs/source/harness_trainer.rst`) and `AGENTS.md`
  §5a get a short "Preflight" paragraph.

### Why this design

- It exercises the **real** wiring: the harness's builders, `epoch_fn`,
  `epoch_kwargs`, and loss, plus the dataset-model-loss-datastep contract.
  These are where mismatches hide. A separate checker script would re-create
  that wiring and could drift from it.
- It needs no changes to existing harness scripts.
- Rejected alternative: a standalone `yoke-preflight` CLI that imports the
  harness module. It can't get `epoch_fn`/`epoch_kwargs`, because they are
  built inside each script's `__main__` block.

### Risks

- `gloo` initialization in tests under `-Werror` might emit warnings on some
  platforms. If so, scope a filter in the test module (the pattern already
  used in `tests/utils/training/epoch/test_loderunner_epoch.py`), not
  globally.
- Large models can be slow or memory-hungry on a CPU. The launch skill
  reports the parameter count first and asks before running.
- Login nodes may forbid heavy compute. The launch skill asks the user where
  preflight should run. If not on the login node, it gives a ready-made
  command for an interactive or debug allocation **for the user to run**.
- Harnesses that bypass the eval datastep's contract (bespoke epoch
  wrappers such as `ch_DDP_diffLDR`'s `run_epoch`) still work, because
  preflight calls whatever `epoch_fn` the harness passed.

## Code change: move the loss to the device (known issue 7)

### What

In `HarnessTrainer.setup()` (`src/yoke/harnesses/trainer.py:294`):

```python
self.loss_fn = self.loss_builder().to(self.device)
```

### Why

A loss with registered buffers or parameters (for example
`MaskedLossMultiplicative`'s `mask`) otherwise stays on the CPU while the
predictions are on the GPU, and the first step fails with a device mismatch.
Losses without buffers are unaffected (`.to()` is a no-op for them), so no
existing harness changes behaviour.

### Notes

- `loss_builder` may return a plain callable rather than an `nn.Module`
  (the type says `nn.Module`, but a harness could pass a function). Guard
  with `if isinstance(loss, nn.Module)` so plain callables still work.
- The preflight change makes the device `cpu`, so this is exercised by the
  preflight tests as well.

### Tests

In `tests/harnesses/` (next to the existing `HarnessTrainer` tests): a loss
module with a buffer ends up on `trainer.device` after `setup()`; a plain
callable loss is left as is. Device moves are tested with `cpu` and, where
needed, the `meta` device, so no GPU is required.

### Follow-ups

- Mark issue 7 as fixed in `dev_plans/known_issues_from_skills_work.md`.
- Simplify the `yoke-new-loss` wiring guidance (no device workaround needed).

## Docs page: "Using Yoke with an AI agent"

### What

A new page `docs/source/agent_skills.rst`, added to the "Concepts" toctree in
`docs/source/index.rst`, and a short "Using Yoke with an AI agent" section in
`README.md` pointing to it.

### Audience

Someone who has data and wants to train a model with Yoke, and does **not**
know Yoke's internals. It is not a developer reference; `skills/README.md`
and `AGENTS.md` §6a stay the maintainer-facing descriptions.

### Contents (short; one screen or two)

1. **What the skills do**: a guided conversation that takes you from "I
   have data" to a training run you submit yourself, and helps debug it
   afterwards.
2. **What you need**: a Yoke install and its Python path; one or two sample
   data files; the cluster you'll train on; Claude Code (or another agent
   that can read `AGENTS.md`).
3. **How to start**: open the agent in the Yoke repo and say what you want,
   for example "I want to train a model on my data in `/path`", or run
   `/yoke-train-my-data`. Mention the other skills by name for users who
   want just one step.
4. **What to expect**:
   - questions in small rounds;
   - decisions recorded in `dev_plans/<project>_training.md`;
   - approval before any code is written;
   - the agent never submits jobs: it gives you the exact command;
   - come back any time later and ask it to look at a run
     (`yoke-run-doctor`).
5. **What the agent will and won't do**: no job submission, no package
   installs, no commits, read-only inspection of data and the cluster.
6. **Where things end up**: new dataset, model, and loss code in
   `src/yoke/`, the harness in `applications/harnesses/<name>/`, run outputs
   in `runs/study_NNN/`.
7. **Limits**: generated code still needs your review; first runs should use
   the smoke CSV.

### Checks

`sphinx-build -b html docs/source _build` builds without new warnings (the
CI "Build documentation" step runs this). The page names only skills and
commands that exist when it is written.

## Skill: `yoke-train-my-data` (front door)

1. **Frame the problem** (round 1): what the data is, what to predict from
   what, how success will be judged (a metric, a visual check, a downstream
   use), the size of the data, the compute available (machine, GPU hours),
   and the deadline. No Yoke terms needed from the user.
2. **Translate** the answers into Yoke terms and show them: task type
   (surrogate, inverse, forward, reward/policy, other); a candidate model
   family; a candidate batch contract; which skills will be needed.
3. **Create the progress file** `dev_plans/<project>_training.md`. It holds a
   checklist of stages (dataset, model, loss, training step, harness, SLURM,
   launch), each stage's status, links to its notes file, and the collected
   cards. Every phase 2 skill reads and updates it.
4. **Route** to each needed skill in order, skipping stages that existing
   Yoke parts already cover (an existing dataset, model, or loss). After each
   one, update the progress file and summarize the decisions so far in plain
   language.
5. **Resume**: if the progress file exists, read it and continue from the
   first unfinished stage. This makes multi-session work safe.
6. **Teach as it goes**: when a Yoke concept first comes up (batch contract,
   harness, study CSV, continuation), explain it in two sentences, with a
   pointer to `skills/README.md` or `docs/`. The user won't know the
   internals.

`yoke-new-harness` stays usable on its own, but routing moves to the front
door. Its "Orient" step will point there for users starting from scratch.

## Skill: `yoke-model` (select or compose)

1. **Requirements** (round 1): input and output tensors from the Dataset
   card (shapes, channels, frames); the task; data size; GPU memory per
   device; any architecture the user prefers or must reproduce.
2. **Select**: check the catalog (`model_catalog.md`, moved from
   `yoke-new-harness/reference/` into this skill; references updated). For
   each candidate, check:
   - the forward signature against the batch contract;
   - the constraints (`validate_patch_and_window`, `embed_dim ==
     num_attention_heads * attention_head_dim`, `num_input_frames`, image
     divisibility);
   - parameter count and a rough memory estimate.

   Propose one or two, with a starting size.
3. **Compose**, only if no existing model fits. A new module in
   `src/yoke/models/`, built from the cataloged blocks (new reference file
   `building_blocks.md`: each block's signature, input/output shapes, and an
   example of use from existing models). The forward signature must match
   the batch contract's model call, or the user must accept a new training
   step.
4. **Design and approval**: class, constructor arguments (plain types only,
   because `model_args` is saved in checkpoints and rebuilt from
   `make_model_args`), forward signature, shape walk-through, parameter count.
5. **Implement and test** (for a composed model): `tests/models/test_<module>.py`
   covering output shape on a tiny input, a finite backward pass, rebuild
   from `model_args` (checkpoint round-trip through
   `yoke.utils.checkpointing`), constraint errors, and parameter count for a
   reference configuration.
6. **Model card** (new in `cards.md`): class, `make_model_args` sketch,
   forward call, constraints, parameter count, memory note.

## Skill: `yoke-new-loss`

1. **Requirements**: what "good" means for the predictions (pointwise error,
   relative error, per-channel balance, masked regions, physical
   constraints); the scale differences between channels; and the loss shape
   the chosen training step requires (from `batch_contracts.md` or the Step
   card).
2. **Select** from a loss catalog (new `reference/loss_catalog.md`): each
   existing loss with its output shape for each `reduction`, its caveats
   (`NormalizedMSELoss` uses batch statistics, so results change with batch
   size), and which training steps accept it.
3. **Compose or write**: prefer wrapping (mask or crop wrappers, a weighted
   sum of existing losses) over new maths. A new loss goes in
   `src/yoke/losses/<name>.py`, as an `nn.Module` with `reduction="none"`
   returning the shape the training step needs.
4. **Harness wiring**: `loss_builder` takes no arguments, so show the
   closure or `functools.partial` pattern for parameterized losses. With the
   issue 7 fix in this phase, the trainer moves the loss to the device, so
   tensors a loss needs (masks, weights) should be registered buffers
   (`register_buffer`), not plain attributes, so that `.to()` moves them.
5. **Tests** (`tests/losses/test_<name>.py`): values against a hand
   computation; output shape for each reduction; finite gradients; edge
   cases (all zeros, eps paths); a wrong-shape input error. GPU behaviour is
   not tested.
6. **Loss card** (new): class, constructor arguments, output shape, the
   `loss_builder` snippet, and which training steps it is compatible with.

## Skill: `yoke-launch-study`

Run on the cluster login node, after `yoke-new-harness`. It **never
submits**.

1. **Read** the progress file and the harness's notes and cards.
2. **Environment checks** (read-only): the user's Python imports `yoke` and
   `torch`; `yoke-start-study` is on `PATH`; the data and filelist paths in
   `training_input.tmpl` exist; the filelists are non-empty and their first
   few entries resolve to files; the CSV renders (`--dryrun` into a
   temporary directory, as in the harness skill).
3. **Preflight**: report the parameter count. Ask where to run it: the login
   node (small models) or an interactive or debug allocation (the skill
   prints the `salloc`/`srun` line for the **user** to run, and the user
   pastes the output back). Command:
   `python <train_script> @<rendered START.input> --preflight`, run from the
   temporary render directory. Interpret the report.
4. **Readiness summary**: total GPU hours (rows × nodes × GPUs × wall time ×
   cycles); the expected output files; how to watch the run
   (`squeue -u $USER`, `tail -f runs/study_NNN/*.out`, `yoke-plot-loss`).
5. **Print the exact launch command** for the user to run from the harness
   directory:

   ```bash
   cd applications/harnesses/<name>
   yoke-start-study --csv <study.csv> --rundir ./runs --cpFile cp_files.txt \
       --submissionType slurm
   ```

   Recommend starting with the smoke CSV.
6. **Write the Run card** (new) into the progress file: harness dir, CSV,
   run dir, study IDs, the command, the date prepared, and what to tell
   `yoke-run-doctor` later.

## Skill: `yoke-run-doctor` (analyze/debug a submitted run)

Starts in **any** session, from the progress file or the Run card, or simply
from "look at `runs/study_003`". Read-only towards SLURM and run outputs.
It never cancels, resubmits, or deletes, and never writes inside `runs/`.

1. **Locate**: the harness dir, study dirs, and expected files (from the Run
   card, or by listing the run directory).
2. **Gather** (read-only):
   - `sacct -j <ids> --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,NodeList`;
     job IDs come from the `.out` files or `sacct -u $USER --name=<job prefix>`;
   - `squeue -u $USER`;
   - the tail of each cycle's `.out`/`.err` file;
   - the chain of `studyNNN_restart_training_epochMMMM.*` files and
     checkpoints;
   - the record CSVs (with `pandas`, reading only what is needed).
3. **Diagnose** against a catalog of failure signatures (new
   `reference/failure_signatures.md`): CUDA out of memory; NCCL timeouts or
   interface errors (`NCCL_SOCKET_IFNAME`); `TIMEOUT` before the checkpoint
   was written (cycle too long for the wall time); the resubmission chain
   stopping (no restart file, `sbatch` error in `.out`); missing data files
   or a filelist error; NaN or exploding loss; loss stuck flat (LR,
   normalization); training/validation divergence (overfitting); a scheduler
   warning; an import or environment error.
4. **Analyze progress**: summarize the loss curves (per study, per dataset tag
   if present), suggest a `yoke-plot-loss` command, and compare studies in
   the sweep.
5. **Recommend**, as concrete proposals: CSV or template edits, a restart
   from the last checkpoint (the exact command, for the **user** to run),
   or code fixes routed to the right skill. Edits to harness files follow
   the approval gate.
6. **Record** findings in the progress file under the Run card, with the
   date, so later sessions see the history.

## Changes to existing files

- `skills/shared/cards.md`: add Model, Loss, and Run cards.
- `skills/shared/interview_conventions.md`: add the progress-file convention
  and the "user submits" rule.
- `skills/yoke-new-harness/SKILL.md`: add preflight to Verify (using the
  CPU flag on sample data, when the data is available locally); point
  "Orient" to `yoke-train-my-data`; end by offering `yoke-launch-study`.
- `skills/yoke-new-dataset/SKILL.md`: hand off to `yoke-model` /
  `yoke-new-loss` when called from the front door.
- `skills/yoke-new-harness/reference/model_catalog.md` → move to
  `skills/yoke-model/reference/model_catalog.md`, and update references.
- `skills/README.md`, `AGENTS.md` §6a: list the new skills and the flow.
- `.claude/skills/`: five new symlinks.
- `tests/skills/test_skill_files.py`: add the five names to
  `EXPECTED_SKILLS`. The existing checks then cover the new files
  automatically.
- `dev_plans/known_issues_from_skills_work.md`: add issue 7 (the loss is not
  moved to the device), then mark it fixed once the change lands.
- `src/yoke/harnesses/trainer.py`: move the loss to the device (issue 7), with
  a test in `tests/harnesses/`.
- `docs/source/agent_skills.rst` (new), `docs/source/index.rst` (toctree
  entry), `README.md` (short pointer section).
- `src/yoke/helpers/cli.py`, `src/yoke/harnesses/trainer.py`,
  `src/yoke/utils/parallel.py`: the preflight change, with tests in
  `tests/harnesses/` and `tests/utils/`.

## Steps (implementation order)

1. **Preflight code**: `setup_single_process_cpu`, CPU-safe DDP wrap,
   `--preflight`, `HarnessTrainer.preflight()`, and tests. A test harness
   uses a tiny dataset and model from the test file, run through the real
   `train_DDP_array_epoch` and `train_DDP_loderunner_epoch` paths. Docs
   paragraph. It comes first because two skills depend on it.
2. **Issue 7 fix**: move the loss to the device in `HarnessTrainer.setup()`,
   with tests; update the known-issues file.
3. Shared updates (cards, conventions).
4. `yoke-model` (move the catalog; write `building_blocks.md` from the source
   of each block).
5. `yoke-new-loss` (`loss_catalog.md` from the source).
6. `yoke-launch-study`.
7. `yoke-run-doctor` (`failure_signatures.md`; real signatures to be filled
   in from your HPC test runs where possible).
8. `yoke-train-my-data`.
9. Updates to the phase 1 skills, `skills/README.md`, `AGENTS.md`, symlinks,
   `EXPECTED_SKILLS`.
10. Docs page `docs/source/agent_skills.rst`, written last so it describes
    the skills as built; `index.rst` and `README.md` pointers.
11. Verification (below), then rehearsal.

## Verification

- **Preflight code**: new unit tests pass with `pytest -Werror`, the full
  suite passes, and `ruff` is clean on changed files (`src/yoke/helpers` is
  lint-blocking in CI). Run `--preflight` on `ch_ldrViT_lsc_cyl` against
  `data_subsamples/` (a CPU forward pass of the real model; the parameter
  count is checked first, and I'll ask before running if it is large).
- **Issue 7 fix**: the new `tests/harnesses/` tests pass, and the existing
  `HarnessTrainer` tests still pass.
- **Docs page**: `sphinx-build -b html docs/source _build` builds, with no new
  warnings from the page.
- **Skills**: `pytest -Werror tests/skills` (frontmatter, symlinks, paths,
  `yoke.` names, contracts), plus the deliberate-breakage check from phase 1
  for at least one new file.
- **Rehearsals** (by you, in fresh sessions):
  - the front door end to end on `data_subsamples/`, as a user who doesn't
    know Yoke;
  - `yoke-launch-study` on the HPC machine, then you submit the smoke CSV;
  - `yoke-run-doctor` in a **new** session the next day, on that run, and on
    one deliberately broken run (for example, wall time too short for a cycle).
- **Not verifiable here**: anything on the cluster, GPU or NCCL behaviour, and
  real failure output, which `failure_signatures.md` should be refined from.

## Open questions

None at present. Resolved in Rev 2: CPU preflight on login nodes assumed OK
(revisit after HPC testing); catalog move approved; issue 7 fix and the docs
page are in scope.

## Later (separate plan)

Making harnesses config-driven, so a new harness is mostly a config file:
see `dev_plans/config_driven_harness_idea.md`. To start after this phase is
done and tested.

## Implementation status (2026-11-11)

Steps 1–10 are implemented. Step 11 (rehearsals) is **not done**: it needs the
user, a fresh session, and the HPC machine.

Changes from the plan:

- **Extra bug fix (known issue 8):** `make_distributed_dataloader` always
  passed `prefetch_factor=2`, which `DataLoader` rejects when `num_workers=0`.
  So preflight, or any `--num_workers 0` run, crashed. It now passes `None`
  when there are no workers. Test in `tests/test_training_utils.py`.
- `preflight()` returns its report dict (handy for tests) and raises
  `RuntimeError` if no validation loss was recorded, not only on a non-finite
  loss.
- The DDP wrap passes `device_ids` only on CUDA, which is needed for CPU DDP.
- `setup_single_process_cpu` asks the OS for a free port.
- The loss device move is guarded with `isinstance(loss, nn.Module)`, as
  planned.
- `yoke-new-loss/SKILL.md` referenced a removed plan
  (`lsc_cyl_joint_ldrViT_harness.md`); the skills CI path check caught it, and
  the text now points to the harness script's docstring.
- `skills/yoke-model/reference/model_catalog.md` was moved with `git mv`, so
  the rename is **staged** in the index (not committed).

Verified:

- `pytest -Werror` full suite: 836 passed, 1 skipped.
- New `tests/harnesses/test_trainer_preflight.py` (11 tests). These use a real
  one-process `gloo` group, real DDP, real distributed dataloaders, and the real
  `train_DDP_array_epoch` / `train_DDP_loderunner_epoch`. They cover a
  contract-mismatch error, a NaN loss, no optimizer step, and the loss device
  move.
- `--preflight` on the real `ch_ldrViT_lsc_cyl` harness against
  `data_subsamples/` (a small 0.9M-parameter configuration on CPU; the frame
  draw was pinned to the sample frames that exist, through a wrapper script in a
  temp dir). It passed: batch `(1, 2, 20, 1120, 400)`, loss 451.2, 2.5 s.
- `pytest -Werror tests/skills`: 110 passed, 1 skipped. A deliberately broken
  copy of a new skill file failed as expected (a misspelled `yoke.` name and a
  missing symlink) and passed again after restoring it.
- `ruff check`, `--preview`, and `format --check` are clean on all changed
  Python files.

Not verified:

- **Sphinx build**: `sphinx` and `furo` are not installed in the local venv, so
  the docs pages were not built here. `agent_skills.rst` parses cleanly with
  docutils. CI's "Build documentation" step will build them.
- Preflight at full model size (327M parameters for the smoke-CSV row) was not
  run on this laptop.
- `failure_signatures.md` is based on general experience. Refine it from real
  failures on LANL machines.
- All cluster behaviour.

## Rev 3: optional agent submission (implemented 2026-11-11)

### Change

After the readiness checks and preflight, `yoke-launch-study` still prints
the exact `yoke-start-study` command. It then **asks** whether the user wants
the agent to run it. If the user says yes, the agent runs it. If not, the user
runs it, as before.

### Why this is safe for SLURM

`yoke-start-study --submissionType slurm` runs `sbatch` once per CSV row
(`HarnessStudy.submit_job`, `src/yoke/harnesses/base.py:305-320`) and returns.
The jobs then live in the SLURM queue, and continuation resubmits from the
compute nodes (`HarnessStudy.continuation_setup`). Nothing depends on the
shell that ran `yoke-start-study`, so the session can end at once. The Rev 1
rule ("the user always submits") was stricter than needed.

### Limits

- **SLURM only.** With `--submissionType shell`, `submit_job` runs the job
  script with `source` (`base.py:63`, `:320`). The training then runs inside
  the shell that called it: here, the agent's tool shell, which would block
  and could be killed with the session. For shell studies the skill gives the
  command only, and says why.
- **Explicit consent at launch time**, for each submission. Approving the
  design or the preflight is not consent to submit. The skill restates the
  exact command, the CSV rows (study IDs), and the GPU-hour estimate, then
  asks. Recommend the smoke CSV for the first submission.
- **Submit only `yoke-start-study`.** The agent runs no other `sbatch`,
  `salloc`, or `srun`, and never `scancel` or `scontrol update`. Other
  skills keep the no-submit rule.
- **Existing study directories.** Before submitting, check that
  `runs/study_NNN/` does not already exist for the CSV's rows (relaunching
  would overwrite rendered files and mix outputs). If any exist, stop and ask.
- **After submitting:** capture the job IDs from `squeue -u $USER` (or
  `sacct`) filtered by the job-name prefix, and fill in the Run card's
  `submitted:` field (date, job IDs, who submitted). Show `squeue` output
  once. Don't keep polling; point to `yoke-run-doctor` for later.

### Files to change

- `skills/yoke-launch-study/SKILL.md`: the description line; the intro (the
  user may choose to have the agent submit); a new step 5b, "Offer to
  submit", with the checks above; the Run card step records who submitted.
- `skills/yoke-launch-study/reference/launch_checks.md`: add "7. Submitting"
  (study-dir check, the command, capturing job IDs).
- `skills/shared/interview_conventions.md`: the safety rule becomes "Never
  submit, cancel, or modify jobs, **except** that `yoke-launch-study` may run
  `yoke-start-study` for a SLURM study when the user explicitly asks at
  launch time."
- `skills/shared/cards.md`: Run card `submitted:` notes "by user / by agent
  at user's request", with the job IDs.
- `skills/README.md` (ground rules, flow diagram), `AGENTS.md` §6a,
  `README.md` and `docs/source/agent_skills.rst` ("You submit the job" becomes
  "you submit, or ask the agent to"), and `skills/yoke-train-my-data/SKILL.md`
  step 5.
- `skills/yoke-run-doctor/SKILL.md`: unchanged (still read-only; it proposes
  restart commands for the user). See the open question.

### Verification

`pytest -Werror tests/skills` (frontmatter and paths). It is a docs-only
change, so there is no new code test. A rehearsal on HPC: let the agent
submit the smoke CSV, then end the session and check with `squeue` from a new
shell that the jobs are still queued or running.

### Decision: `yoke-run-doctor` stays propose-only

Decided with the user (2026-11-11): `yoke-run-doctor` does **not** submit
anything, including existing restart scripts. Resubmitting a broken chain is
riskier, because it can duplicate a cycle or resume from the wrong checkpoint.
Revisit after the HPC tests show how chains actually break.
