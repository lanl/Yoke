DiffusionLodeRunner Training - DDP - Chicoma
============================================

Chicoma (`ch_`) DDP harness for training the **DiffusionLodeRunner** model — a
score-based diffusion variant of LodeRunner — on the `lsc240420` dataset. Uses
PyTorch `DistributedDataParallel` with a VP cosine noise schedule.

Files
-----

- `train_DDP_diffLDR.py` — DDP training script for DiffusionLodeRunner.
- `training_input.tmpl` — single input template (with the
  `# <<optional:CONTINUATION>>` block for continuation runs). `<CHECKPOINT>` is a
  reserved placeholder filled in by the continuation logic on resume; it is not a
  CSV column.
- `training_slurm.tmpl` — complete SLURM submission script (Venado GPU partition).
- `cp_files.txt` — files copied into each `study_###` run directory.
- `study_template.csv` — hyperparameters for three model-size studies.

Study
-----

`study_template.csv` defines three studies that vary the LodeRunner model shape:
embedding dimension (`EMBED_DIM`) and Swin block structure (`B0`–`B3`), along with
batch sizing and node/GPU counts. `MAX_TIME_OFFSET` controls the temporal offset
between the input and predicted frames.

Launch
------

From within this directory, with the Yoke environment active:

```bash
yoke-start-study --csv study_template.csv --submissionType slurm
```

Add `--dryrun` to render the study directories and print the `sbatch` commands
without submitting any jobs.
