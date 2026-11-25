# Launch Checks

Checks to run on the cluster login node before giving the user the launch
command, in this order. Checks 1–6 are read-only, apart from the one-batch CPU
preflight. Check 7 submits, and only on the user's explicit yes. Show each
command before running it. Stop and report at the first failure. Do not work
around it silently.

`<python>` is the interpreter the user named. `<harness>` is
`applications/harnesses/<name>`.

## 1. Environment

```bash
<python> -c "import yoke, torch; print(yoke.__file__, torch.__version__)"
command -v yoke-start-study yoke-plot-loss
```

- `yoke` must import from the checkout the user will launch from (an
  editable install points into it), or the study will run different code.
- `yoke-start-study` must be on `PATH` in the shell that will submit:
  the agent's, if the user asks the agent to submit, or the user's own,
  which may differ.
- The SLURM template's activation line (`source .../activate` or
  `conda activate ...`) must name the same environment.

## 2. Templates render

From `<harness>`, into a temporary directory (never `./runs`):

```bash
T=$(mktemp -d)
yoke-start-study --csv <smoke.csv> --rundir $T --cpFile cp_files.txt \
    --submissionType slurm --dryrun
grep -nE "<[A-Za-z_]+>" $T/study_*/study*_START.*    # must print nothing
```

## 3. Data and filelists exist

Read the paths from a rendered `$T/study_001/study001_START.input`. For each
directory or file argument:

```bash
test -d <dir> && echo ok || echo MISSING
wc -l <FILELIST_DIR>/<filelist>        # non-empty
head -3 <FILELIST_DIR>/<filelist>
```

For the first three entries of each filelist, check that a matching data file
exists. Use the pattern the dataset uses, for example
`ls <NPZ_DIR>/<prefix>_pvi_idx00000.npz`. Report how many were found. Do not
scan whole directories on shared file systems.

## 4. Preflight (one CPU forward pass)

1. Report the parameter count first. Build the model on the CPU from the
   parsed args, or take it from the Model card. For more than about 500M
   parameters, warn that a CPU forward pass may take minutes and several GB,
   and ask before going on.
2. Ask where to run it:
   - **login node** (default for moderate models; site policy permitting):
     run it as below;
   - **interactive or debug allocation**: print the allocation command for
     the **user** to run (for example
     `salloc -p <debug partition> -N 1 -t 00:30:00 -A <account>`, from the
     machine profile). The user runs the preflight inside it and pastes the
     output back. The agent never runs `salloc`/`srun`.
3. Run from the rendered study directory, so relative paths match a real
   job:

   ```bash
   cd $T/study_001
   <python> <train_script> @study001_START.input --preflight
   ```

4. Read the report:
   - `batch:` shapes and dtypes match the Dataset card;
   - `parameters:` matches the Model card;
   - `losses:` finite and of a plausible size (for `LpLoss` this is much
     larger than an MSE);
   - `Preflight passed.` printed, exit status 0.
5. On failure, map the error to a cause:

   | Error | Likely cause | Go to |
   | --- | --- | --- |
   | `ValueError: not enough values to unpack` / `too many values` | dataset tuple ≠ datastep contract | `yoke-new-dataset` or `yoke-new-training-step` |
   | shape mismatch in `forward` | model args vs. data shape or channels | `yoke-model` |
   | `ValueError` about loss shape `(B, C)` | wrong loss for the datastep | `yoke-new-loss` |
   | `FileNotFoundError` / dataset `RuntimeError: no usable sample` | paths, filelists, missing frames | check 3 above |
   | `Preflight loss is not finite` | NaNs in the data, `eps`, normalization | `yoke-new-dataset` (NaN handling) |
   | `Unsupported dataset` | wrong `dataset` tag in `epoch_kwargs` | harness script |

## 5. Budget

GPU hours ≈ Σ over CSV rows of
`KNODES × NGPUS × wall time per cycle × ceil(total_epochs / cycle_epochs)`.

CPU mode (`device_type="cpu"`, one node per job): node hours ≈ Σ over CSV rows
of `wall time per cycle × ceil(total_epochs / cycle_epochs)`. On partitions that
allocate whole nodes, the whole node is charged even if few threads are used.
This is an upper bound: a cycle that finishes early uses less. Give the smoke
CSV and the production CSV separately.

## 6. Clean up

`rm -rf $T` (it holds only rendered files). Never remove anything under the
harness's own `runs/`.

## 7. Submitting (only on the user's explicit yes; SLURM only)

Before:

```bash
cd <harness>
# Every study ID in the CSV, then any matching run directory that exists.
<python> -c "import pandas as pd; \
print(*pd.read_csv('<study.csv>', index_col=0, comment='#').index)"
ls -d runs/study_* 2>/dev/null
```

If any `runs/study_NNN` matches a CSV row, stop and ask (see step 5 of
`SKILL.md`). Confirm that the submission type is `slurm`.

Submit, once, exactly as shown to the user:

```bash
cd <harness>
yoke-start-study --csv <study.csv> --rundir ./runs --cpFile cp_files.txt \
    --submissionType slurm
```

After:

- Expect one `Submitted batch job <id>` line per CSV row (from `sbatch`) and
  `[COPY]` lines. Any `sbatch: error` means that row was not submitted:
  report it and don't retry.
- `squeue -u $USER --name=<job prefix>_s<NNN>_e0001` (or `squeue -u $USER`)
  once, to show the jobs are queued.
- Record the job IDs and the date in the Run card's `submitted:` field.
- Don't poll the queue. Later status checks are `yoke-run-doctor`'s job.
