---
name: yoke-new-training-step
description: Interview the user and write a new train/eval datastep pair and DDP epoch function (or a new DATASTEP_FN entry) in src/yoke/utils/training/, with unit tests, for a batch contract or loss/recording behaviour that no existing Yoke training step supports. Use when a dataset's batch tuple matches no row of skills/shared/batch_contracts.md, or when the user needs a new training-step behaviour.
---

# Create a Yoke Training Step (Datastep + Epoch Function)

You guide the user through designing a training step for a batch format or
behaviour Yoke lacks, then implement it with unit tests. The result must
plug into `HarnessTrainer` unchanged.

Read first:

- `skills/shared/interview_conventions.md` (follow it throughout).
- `skills/shared/batch_contracts.md`.
- `skills/yoke-new-training-step/reference/datastep_epoch_anatomy.md`.
- The Dataset card, if one exists (from `yoke-new-dataset`, in its
  `dev_plans/` notes).

Load when implementing tests:

- `skills/yoke-new-training-step/reference/training_step_test_patterns.md`
- `skills/shared/cards.md`

## Procedure

### 1. Orient

- Ask which Python environment has Yoke installed.
- Find the closest existing pair in `batch_contracts.md`. Open its source,
  then tell the user which it is and exactly how the new case differs (tuple,
  model call, loss shape, records).
- Confirm that a new step is really needed. If a small change to the
  dataset's output would fit an existing contract, say so and let the user
  choose.
- Create the design notes file `dev_plans/<family>_training_step.md`
  (or add to the dataset's notes file if the user prefers).

### 2. Interview (rounds)

Round 1, data and model:

- The batch tuple (from the Dataset card): names, shapes with `B`, dtypes.
- The model and its exact forward call (see
  `skills/yoke-model/reference/model_catalog.md`). Which tuple elements
  feed it, and which are targets or metadata?
- Fixed channels (`channel_map` argument) or per-batch channels?

Round 2, loss and recording:

- Which loss, and what shape it returns: elementwise, per-channel `(B, C)`,
  or per-sample.
- What is backpropagated (usually the mean over everything) and what single
  number per sample is recorded.
- What should happen when the loss has the wrong shape (recommended: raise
  `ValueError`).
- Extra record columns (for example a dataset tag) and whether
  `yoke-plot-loss` must read them.

Round 3, loop behaviour:

- EMA (`ema_model`, `global_step`, `ema_update_after_step`) and `grad_clip`
  support. Recommended: yes, matching `train_DDP_loderunner_epoch`.
- Anything else per batch: scheduled sampling, multi-step rollout, auxiliary
  losses.
- What the eval step records. Whether a non-DDP eval step is needed for
  post-training evaluation scripts.

Then choose the shape of the change (see the anatomy file): a new
`DATASTEP_FN` entry, a new epoch function in an existing module, or a new
module. Recommend the least new code.

### 3. Design and approval

Write in the notes file:

- Function names (`train_DDP_<family>_datastep`, `eval_DDP_<family>_datastep`,
  `train_DDP_<family>_epoch` or the `DATASTEP_FN` tag), full signatures, and
  target files.
- The batch contract, model call, loss shape check, reductions, and record
  format.
- Changes to `DATASTEP_FN` and `__all__`, if any.
- New rows for `skills/shared/batch_contracts.md`.
- The test plan (from `training_step_test_patterns.md`) and verification
  commands.

Show it and **wait for approval**.

### 4. Implement

- Datasteps and epoch function following `datastep_epoch_anatomy.md`:
  unpack `data` in one line, check the loss shape, gather on all ranks, and
  write records on rank 0. Use Google docstrings with `Args`/`Returns`/
  `Raises`, full annotations, and lines of 89 characters or fewer.
- Tests under `tests/utils/training/datastep/` and
  `tests/utils/training/epoch/`, following
  `training_step_test_patterns.md`.
- Add the new rows to the table in `skills/shared/batch_contracts.md`
  (between the `contracts:begin`/`contracts:end` markers) and a short
  section under "Contracts in detail".

### 5. Verify

- `pytest -Werror tests/utils/training` (new and neighbouring tests, to
  catch regressions in shared tables).
- `pytest -Werror tests/skills` (checks the contract table against the new
  code).
- `ruff check`, `ruff check --preview`, `ruff format --check --diff` on the
  changed files.
- Report results honestly. Say that this checks the logic only: no
  multi-rank or GPU run was made.

### 6. Hand off

Write the **Step card** (`skills/shared/cards.md`) in the notes file, show
it, and update the progress file if there is one. Then return to the
calling skill, or offer `yoke-new-harness`.
