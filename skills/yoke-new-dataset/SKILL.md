---
name: yoke-new-dataset
description: Interview the user about a new dataset (any file format - npz, HDF5, CSV/txt/dat, TIFF/images, binary) and its learning task, then write a torch Dataset class in src/yoke/datasets/ with tests that matches an existing Yoke batch contract. Use when the user wants to train a Yoke model on data Yoke cannot load yet, or asks to add/create a dataset class.
---

# Create a Yoke Dataset Class

You guide the user through designing a `torch.utils.data.Dataset` for their
data, then implement and test it. The dataset must return a batch tuple
that a Yoke training step understands (a **batch contract**). Ask questions;
do not guess facts the user or the data can provide.

Read first:

- `skills/shared/interview_conventions.md` (how to run the interview, safety
  rules, approval gate). Follow it throughout.
- `skills/shared/batch_contracts.md` (the contracts).
- `AGENTS.md`.

Load when the procedure says to:

- `skills/yoke-new-dataset/reference/format_inspection.md`
- `skills/yoke-new-dataset/reference/dataset_patterns.md`
- `skills/yoke-new-dataset/reference/dataset_test_patterns.md`
- `skills/shared/cards.md`

## Procedure

### 1. Orient

- Ask which Python environment has Yoke installed.
- List `src/yoke/datasets/` and skim the module docstrings. Ask what the
  data is and what it will be used for, in a sentence or two.
- If an existing dataset is close, say which and why, and ask whether to
  subclass or extend it rather than write a new module.
- Create the design notes file `dev_plans/<short_name>_dataset.md`.

### 2. Data inventory (round 1)

Ask:

- File format(s), and how they are produced (simulation code, instrument,
  preprocessing script).
- Directory layout and filename pattern: one file per sample, per time step,
  per simulation, or one big file. Give an example file name.
- Where one or two sample files are (for example `data_subsamples/`) that
  you may inspect.
- Any metadata or design table (CSV) and what keys it uses.
- Known gaps, corrupt files, or bad samples.
- Rough size per file and in total, and the number of simulations or
  samples.

### 3. Inspect samples (cheap, read-only)

Use the recipe for the format in `format_inspection.md`. Report the
names, shapes, dtypes, NaN or sentinel use, value ranges on a slice, scalar
metadata, and the filename encoding. Ask the user to confirm or correct your
reading. Note any reader that is not a declared dependency (see that file),
and ask before relying on it in `src/yoke`.

### 4. Learning task (round 2)

Ask:

- What is the input and what is the target, in terms of the fields you
  found?
- Time structure: no time, one step ahead, two input frames, a sequence. Is
  the lead time from a stored time field or from index spacing?
- Which model family is intended (see
  `skills/yoke-model/reference/model_catalog.md`)?

Then **propose a batch contract** from `batch_contracts.md`: the tuple, shapes,
dtypes, and the epoch function it implies. Explain why it fits. If none fits:

- First see whether a small change of output makes one fit (for example
  packing extra inputs into `inpt` for the array contract). Offer it.
- Otherwise, record the new tuple as contract `NEW` and tell the user the
  next step will be `yoke-new-training-step`.

### 5. Sampling and splits (round 3)

Read `dataset_patterns.md`, then ask:

- Fixed-index (one item per file) or random-sampling (draw from
  simulations)? Recommend one based on the task.
- How samples are split into train, validation, and test. Use existing
  filelists in `applications/filelists/`, or make new ones split by
  simulation (recommended).
- For time sampling: the maximum index gap, and what to do with gaps or
  missing frames (redraw, next simulation, fail).

### 6. Channels and preprocessing (round 4)

Ask:

- Which fields become channels, and in what order. Is the channel set the
  same for every sample? If not, a channel map is needed, which forces
  `batch_size=1`.
- NaN and sentinel handling.
- Geometry: half or full (reflected) image, cropping, resizing or padding to
  the model's constraints.
- Normalization: none, an existing stats file in
  `applications/normalization/`, or new stats (computed by a separate
  script, not in the dataset).
- Any per-sample metadata to return (ids, design parameters).

### 7. Design and approval

Write a `## Design` section in the notes file:

- Module and class name. Subclass or new class.
- Constructor signature (every argument with type and default; mark path
  arguments).
- The `__getitem__` tuple with shapes and dtypes, and its contract row.
- `__len__` behaviour, RNG, retry or failure behaviour.
- Reader library, and any new dependency (needs approval).
- Filelists to use or create.
- Test plan: the cases from `dataset_test_patterns.md` that apply.
- `## Verification`: the commands you will run.

Show the design to the user and **wait for approval**.

### 8. Implement

- Write the dataset module following `dataset_patterns.md`: Google
  docstrings, full annotations, lines of 89 characters or fewer, double
  quotes, and comments explaining non-obvious choices.
- Write `tests/datasets/test_<module>.py` following
  `dataset_test_patterns.md`, with synthetic files in the same format.
- New filelists, if approved, go in `applications/filelists/`. Write any
  helper script for them outside `src/yoke`, and show it before running it.

### 9. Verify

- `pytest -Werror tests/datasets/test_<module>.py`
- `ruff check`, `ruff check --preview`, and `ruff format --check --diff` on
  the new files.
- With the user's OK: build the dataset on the real sample files, load one or
  two items, and print shapes, dtypes, and min/max per channel. This is a data
  read only, not training.
- Report results honestly, including failures and anything not checked.

### 10. Hand off

- Write the **Dataset card** (`skills/shared/cards.md`) in the notes file and
  show it.
- If there is a progress file (`yoke-train-my-data`), update it and return
  there. The next stages are usually `yoke-model` and `yoke-new-loss`.
- Otherwise: if the contract is `NEW`, offer `yoke-new-training-step`. If not,
  offer `yoke-model` (if the model isn't chosen yet) or `yoke-new-harness`.
