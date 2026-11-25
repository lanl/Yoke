# What a Yoke Run Leaves on Disk

Where to look when diagnosing a study. Everything is inside
`applications/harnesses/<name>/runs/study_NNN/` (`NNN` = `studyIDX`, zero-padded
to 3 digits). Treat all of it as **read-only**.

## Files, in the order they appear

| File | Written by | Meaning |
| --- | --- | --- |
| `<train_script>`, other `cp_files.txt` entries | `yoke-start-study` | the code this study runs (the study's own copy; edits to the harness dir don't affect it) |
| `training_input.tmpl`, `training_slurm.tmpl` | `yoke-start-study` | continuation templates, CSV keys already filled in |
| `studyNNN_START.input`, `studyNNN_START.slurm` | `yoke-start-study` | first cycle's argument file and job script |
| `studyNNN_epoch0001.out` / `.err` | SLURM, cycle 1 | stdout and stderr of the first job (names from `#SBATCH --output/--error`, using `<epochIDX>`) |
| `training_studyNNN_epochEEEE.csv` | epoch function, rank 0 | training records: `epoch, batch, loss[, dataset]`, one row per sample |
| `validation_studyNNN_epochEEEE.csv` | epoch function, rank 0 | validation records, written every `TRAIN_PER_VAL` epochs |
| `studyNNN_modelState_epochEEEE.pth` | `HarnessTrainer.finalize` | checkpoint at the end of a cycle (`EEEE` = last epoch of the cycle) |
| `..._ema.pth`, `..._ema_weights.pth` | EMA hook | EMA companion and production weights, if EMA is on |
| `studyNNN_restart_training_epochMMMM.input` / `.slurm` | `HarnessStudy.continuation_setup` | next cycle (`MMMM` = first epoch of the next cycle), submitted from the compute node |
| `studyNNN_epochMMMM.out` / `.err` | SLURM, later cycles | output of each restarted job |
| `studyNNN_evaluation_<stem>.*`, `testing_...csv` | evaluation (if enabled) | post-training evaluation job and records |

Notes:

- Record names come from the harness's `--trn_rcrd_filename` /
  `--val_rcrd_filename`. The pattern above is the usual one; check
  `training_input.tmpl` for the actual names.
- The first cycle's records all go into the `epoch0001` files (the START
  input already has `<epochIDX>` replaced). Later cycles write one file per
  epoch.
- Rows have no header, except the training records of `train_DDP_array_epoch`
  and `train_lsc_policy_epoch` (`epoch,batch,loss`, repeated once per epoch,
  so a first-cycle file has several). `yoke.plots.loss_curves.load_record_csv`
  skips those header lines and reads the rest with `sep=", "`.

## The healthy chain

For a study with `total_epochs = T` and `cycle_epochs = K`, expect about
`ceil(T / K)` cycles. Each cycle except the last leaves:

1. an `.out`/`.err` pair,
2. a checkpoint `..._epochEEEE.pth`,
3. a `restart_training_epoch{EEEE+1}` `.input`/`.slurm` pair,
4. the next cycle's `.out`.

The last cycle leaves a checkpoint and no restart files (and an evaluation
job, if enabled). **The first missing link is where the problem is.**

## Read-only commands

```bash
ls -lt runs/study_NNN/ | head -30                    # newest first
tail -50 runs/study_NNN/studyNNN_epochEEEE.out
tail -50 runs/study_NNN/studyNNN_epochEEEE.err
grep -iE "error|traceback|killed|oom|nan|timeout|cancel" runs/study_NNN/*.err | tail
squeue -u $USER -o "%.12i %.30j %.8T %.10M %.6D %R"
sacct -u $USER -S <YYYY-MM-DD> --name=<job prefix>_s<NNN>_e* \
      --format=JobID,JobName%30,State,ExitCode,Elapsed,Timelimit,MaxRSS,NodeList
sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed,MaxRSS,MaxVMSize
```

The job-name prefix comes from the SLURM template's
`#SBATCH --job-name=<prefix>_s<studyIDX>_e<epochIDX>` (also in the Run card).
Job IDs often appear near the top of `.out` when the template uses
`#SBATCH -vvv`.

Loss records, reading only what is needed:

```python
from yoke.plots.loss_curves import find_record_csvs, load_record_csv
paths, epochs = find_record_csvs("runs", studyIDX, "training")
df = load_record_csv(paths[-1])          # last epoch only
print(df["Loss"].describe(), df["Loss"].isna().sum())
```

Plots for the user: `yoke-plot-loss --basedir ./runs --IDX <N> [--ylim Y]
[--dataset <tag>] [--inprogress] --savefig`.
