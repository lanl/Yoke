---
name: yoke-model
description: Help the user pick an existing Yoke model (LodeRunner, LodeRunnerViT, CNN surrogates, ...) for their data and task, size it, and check its constraints - or, if none fits, compose a new nn.Module in src/yoke/models/ from Yoke building blocks (patch embeddings, ViT/Swin backbones, CNN modules, MLPs) with shape tests. Use when the user asks which model to use, needs a model for a new dataset, or wants a custom architecture.
---

# Choose or Compose a Yoke Model

You help the user find a model that fits their data and task. Pick an existing
Yoke model when one fits (the usual case). Otherwise build a new `nn.Module`
from Yoke's building blocks. Either way, check the result: input and output
shapes, constraints, and size. The user may not know the architectures, so
explain the trade-offs in plain terms.

Read first:

- `skills/shared/interview_conventions.md` (follow it throughout).
- `skills/yoke-model/reference/model_catalog.md`.
- `skills/shared/batch_contracts.md` (each model's forward call must match a
  datastep).
- The progress file and the Dataset card, if they exist.

Load when composing:

- `skills/yoke-model/reference/building_blocks.md`
- `skills/shared/cards.md`

## Procedure

### 1. Orient

- Ask which Python environment has Yoke installed.
- Read the Dataset card. Write down the input and target tensors (shapes,
  channels, frames, extra inputs such as lead times or vectors) and the batch
  contract.
- Create or extend the design notes file `dev_plans/<project>_model.md`.

### 2. Requirements (round 1)

Ask:

- The task in plain terms: fields to fields (surrogate or time stepping),
  image to numbers (inverse), numbers to image (forward), or a mix.
- Data scale: number of samples or simulations, image size, number of
  channels.
- Compute: GPU type and memory per GPU and number of GPUs, or, on a
  CPU-only machine, cores and memory per node; and the time budget (from the
  machine profile, if there is one). CPU mode (`device_type="cpu"`, one
  process per job) suits small models; check that the size fits a CPU's
  speed before recommending it.
- Any architecture the user prefers or must reproduce, or one that worked
  before.

### 3. Select

Using `model_catalog.md`, propose **one or two** candidates. For each:

- Fit: the forward call matches the batch contract, and inputs and outputs
  match the Dataset card.
- Constraints checked for the **real** sizes:
  - `LodeRunner` / `DiffusionLodeRunner`:
    `yoke.helpers.training_design.validate_patch_and_window` is all `True`.
    If not, propose a patch size or padding
    (`yoke.helpers.training_design.find_valid_pad`, `ResizePadCrop`).
  - `LodeRunnerViT`: `H, W` divisible by `patch_size`,
    `embed_dim == num_attention_heads * attention_head_dim`, an even head
    dimension, `num_input_frames` 1 or 2, and token count
    `N = (H/ph)*(W/pw)` (attention cost grows with `N²`).
  - CNN surrogates: the output size works out (shape walk-through).
- Size: build it on the CPU (with the user's OK) and report the parameter
  count for a small, a medium, and the proposed configuration, plus a rough
  memory note.
- Which arguments go in `make_model_args` as fixed values, which become CLI
  flags, and which become CSV columns for a sweep.

If one fits, skip to step 6 (no new code).

### 4. Compose (only if nothing fits)

Read `building_blocks.md`. Follow its composition rules, and design:

- the block sequence, with the tensor shape after each block for the real
  sizes (base it on the closest existing model's data flow);
- constructor arguments, plain types only (they are saved in checkpoints);
- a forward signature matching a batch contract. Otherwise, plan a new
  training step with `yoke-new-training-step`;
- constraint checks in `__init__`;
- parameter count and memory estimate.

Write this in the notes file and **wait for approval**.

### 5. Implement and test (composed models only)

- `src/yoke/models/<module>.py`: import the blocks, don't copy them. Use a
  Google docstring with the data flow and shapes, full annotations, and lines
  of 89 characters or fewer.
- `tests/models/test_<module>.py` (see the existing tests in `tests/models/`):
  - output shape on a tiny configuration (a small image and patch, a few
    channels);
  - a finite backward pass (`out.sum().backward()`, gradients not `None`);
  - the constraint errors;
  - a rebuild from `model_args`: construct, save with
    `yoke.utils.checkpointing.save_model_and_optimizer`, reload with
    `yoke.utils.checkpointing.load_model_and_optimizer`, and get the same output;
  - the parameter count for a reference configuration (guards against
    accidental changes).
- Verify: `pytest -Werror tests/models/test_<module>.py`, plus `ruff check`,
  `ruff check --preview`, and `ruff format --check --diff`
  (`src/yoke/models` is lint-blocking in CI).

### 6. Hand off

Write the **Model card** (`skills/shared/cards.md`) in the notes file and the
progress file: class, `make_model_args` sketch, forward call, constraints,
parameter count, memory note. Show it and return to the calling skill
(`yoke-train-my-data` or `yoke-new-harness`).
