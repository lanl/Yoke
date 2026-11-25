---
name: yoke-new-loss
description: Help the user choose an existing Yoke loss, wrap or combine existing losses, or write a new loss nn.Module in src/yoke/losses/ with tests, making sure its output shape matches what the chosen Yoke training step requires. Use when the user asks which loss to use, needs a custom/masked/weighted/physics loss, or when a harness needs a loss_builder.
---

# Choose or Create a Yoke Loss

You help the user decide what the training objective should be, then choose,
wrap, or write a loss that returns exactly the shape the chosen training step
needs. Explain the trade-offs in plain terms. The user may not know what
"reduction" or "per-channel" means in Yoke.

Read first:

- `skills/shared/interview_conventions.md` (follow it throughout).
- `skills/yoke-new-loss/reference/loss_catalog.md`.
- `skills/shared/batch_contracts.md` (which loss shape each datastep needs).
- The progress file and any Dataset, Model, or Step cards, if they exist.

## Procedure

### 1. Orient

- Ask which Python environment has Yoke installed.
- Find the training step this loss will feed: from the Step card, the
  harness's `epoch_fn` and `dataset` tag, or the batch contract. Write down
  the **loss shape it requires** (elementwise, per channel `(B, C)`, or per
  sample). Everything below must respect it.
- Create or extend the design notes file
  (`dev_plans/<project>_loss.md`, or a section in the harness notes).

### 2. Interview (rounds)

Round 1, what "good" means:

- What should the model get right: pointwise values, relative error, sharp
  features (shocks, interfaces), integrals or conserved quantities?
- Are some regions irrelevant or invalid (padding, outside the domain, masked
  materials)?
- Do the channels have very different scales (for example densities and
  pressures)? Is the data normalized?

Round 2, constraints:

- Batch size, especially `batch_size=1`, which rules out batch-statistics
  losses such as `NormalizedMSELoss`.
- Anything known from earlier runs. For example, MSE gave poor results for
  2-frame LodeRunnerViT rollouts on raw LSC/CYL fields, and the per-channel
  `LpLoss` largely fixed it (see the module docstring of
  `applications/harnesses/ch_ldrViT_lsc_cyl/train_ldrViT_2frame_lsc_cyl.py`).

### 3. Recommend

In order of preference:

1. **An existing loss** from the catalog, with arguments. For example
   `LpLoss(reduction="none")` for multi-scale fields with a per-channel
   datastep, or `MSELoss(reduction="none")` for normalized data.
2. **A wrapper or combination** of existing losses (mask, crop, weighted
   sum), written as a small `nn.Module` if it must be reused, or as a
   `build_loss()` function in the harness if it is a one-off composition.
3. **A new loss** in `src/yoke/losses/<name>.py`, only when the maths is
   genuinely new.

If the objective needs a shape the training step can't accept (for example a
per-sample scalar for an elementwise datastep), say so. Offer either a
different datastep (`dataset` tag) or `yoke-new-training-step`.

### 4. Design and approval

Write in the notes file: the class or composition; constructor arguments and
chosen values; the formula; the output shape for the chosen reduction; how it
handles `eps`, zeros, and masks; the `loss_builder` snippet; and the test plan.
Show it and **wait for approval**.

### 5. Implement (new or reusable wrapper losses)

- `src/yoke/losses/<name>.py`: an `nn.Module` with a `reduction` argument
  that defaults to `"none"` and returns the shape the training step needs.
  Use Google docstrings with the formula and the output shape for each
  reduction, and full annotations. Register any tensors as buffers.
- `tests/losses/test_<name>.py` (model on `tests/losses/test_lp_loss.py`):
  - values against a hand computation on tiny tensors;
  - output shape for each reduction;
  - finite gradients (`loss.sum().backward()`);
  - edge cases: zeros, identical inputs, `eps` paths;
  - a clear error for invalid arguments or shapes;
  - buffers move with `.to("cpu")` / `.to(torch.float64)`.

### 6. Verify

- `pytest -Werror tests/losses/test_<name>.py`.
- `ruff check`, `ruff check --preview`, and `ruff format --check --diff` on
  the changed files (`src/yoke/losses` is lint-blocking in CI).
- If a harness exists: the harness's `--preflight` (see
  `skills/yoke-launch-study/SKILL.md`, step 3), which runs the loss on a real
  batch.

### 7. Hand off

Write the **Loss card** (`skills/shared/cards.md`) in the notes file and the
progress file, show it, and return to the calling skill (`yoke-train-my-data`
or `yoke-new-harness`).
