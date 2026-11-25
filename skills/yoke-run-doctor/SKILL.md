---
name: yoke-run-doctor
description: Analyze or debug a Yoke training study the user already submitted, in any later session - read SLURM job status (squeue/sacct), the study's .out/.err logs, the checkpoint and restart chain, and the loss records, diagnose failures or poor training, and propose fixes and exact restart commands for the user to run. Read-only; never submits, cancels, or edits run outputs. Use when the user asks why a run failed, stopped, or is training badly, or asks for the status of a study.
---

# Diagnose a Submitted Yoke Run

A Yoke study was submitted earlier (by the user, or by `yoke-launch-study` at
the user's request), and the user wants to
know how it is going or why it went wrong. Work from what is on disk and in
SLURM. Assume nothing is remembered from the session that set it up.

**Read-only.** Never run `sbatch`, `scancel`, `scontrol update`, or
`yoke-start-study`. Never create, edit, or delete files inside `runs/`.
Propose fixes and commands; the user runs them.

Read first:

- `skills/shared/interview_conventions.md` (follow it throughout).
- `skills/yoke-run-doctor/reference/run_layout.md`.
- `skills/yoke-run-doctor/reference/failure_signatures.md`.

## Procedure

### 1. Locate the run

- If there is a progress file with a **Run card** (`dev_plans/*_training.md`,
  or the harness notes file), read it: harness dir, CSV, study IDs, job
  prefix, expected outputs, earlier doctor findings.
- Otherwise, ask for the harness directory (or a `runs/study_NNN` path) and
  infer the rest from the study directory's files (`training_input.tmpl`,
  `training_slurm.tmpl`, the CSV in the harness dir).
- Ask what the user noticed ("it stopped", "loss is NaN", "just checking") and
  which studies to look at.

### 2. Gather evidence (read-only)

Using the commands in `run_layout.md`:

1. List each study directory, newest first, and reconstruct the **chain**:
   cycles with `.out`/`.err`, checkpoints, and restart files.
2. `squeue -u $USER` for running or pending jobs, and `sacct` for this study's
   jobs (by job-name prefix): state, exit code, elapsed time against the limit,
   memory.
3. The tail of the latest `.out` and `.err`, and a grep of all `.err` files for
   errors.
4. Loss records: shape of the curves (start, latest, trend, NaN count) per
   study, and per dataset tag if present. Read the last files, not
   everything.

If you are not on the cluster (no `squeue`/`sacct`), say so and work from
the files the user can make available.

### 3. Diagnose

- Find the **first broken link** in the chain (see "The healthy chain" in
  `run_layout.md`).
- Match the evidence against `failure_signatures.md`. Quote the actual lines
  (error message, `sacct` state, the loss numbers) that support the diagnosis.
- Say how confident you are, and what else could explain it. If nothing
  matches, say so, and list what extra information would help (for example
  `NCCL_DEBUG=INFO` output from one run).

### 4. Report progress (when the run is healthy or still running)

- Epochs done of `total_epochs`, cycles done, time per epoch, and estimated
  time left.
- Training and validation loss trends, and comparison across the studies in
  the sweep.
- A `yoke-plot-loss` command for the user (with a suitable `--ylim`, and
  `--dataset` for tagged records).

### 5. Recommend

Concrete proposals, smallest first:

- CSV or template changes (for example lower `cycle_epochs`, raise `--time`,
  fix the activation line), shown as diffs. Edit harness files only with
  approval. **Never edit the copies inside `runs/`.** A running chain uses
  those copies, so say whether a change needs a fresh study or only affects
  new ones.
- The exact command to resume from the last good checkpoint or restart
  script, **for the user to run** (see "Resubmission chain" in
  `failure_signatures.md`).
- Code problems, routed to the right skill: `yoke-new-dataset`,
  `yoke-model`, `yoke-new-loss`, `yoke-new-training-step`,
  `yoke-slurm-template`. Before resubmitting after a code fix, run
  `yoke-launch-study`'s preflight.

### 6. Record

Add a dated entry under the Run card's `doctor log:` in the progress file
(or the harness notes file): what was checked, the diagnosis, what was
proposed, and what the user decided. If the user tells you job IDs or the
submission date, fill in `submitted:`.

If a real failure differed from `failure_signatures.md`, propose adding it
there, with the actual message, the machine, and the date.
