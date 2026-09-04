LodeRunner Training - DDP - Selene
==================================

Selene (`se_`) variant of the DDP LodeRunner training harness. Uses PyTorch
`DistributedDataParallel` to train the LodeRunner architecture on single-timestep
input and output of the `lsc240420` per-material density fields. This mirrors
`ch_DDP_loderunner` but is tuned for the Selene HPC environment.

Files
-----

- `train_LodeRunner_ddp.py` — DDP training script for LodeRunner.
- `training_input.tmpl` — single input template (with the
  `# <<optional:CONTINUATION>>` block for continuation runs).
- `training_slurm.tmpl` — complete SLURM submission script.
- `cp_files.txt` — files copied into each `study_###` run directory.
- `ddp_paper_study.csv` — hyperparameters for the LodeRunner-18channel paper runs.

Study
-----

`ddp_paper_study.csv` sweeps LodeRunner model size and temporal offset (embedding
dimension `EMBED_DIM`, Swin block structure `B0`–`B3`, max time-index offset
`MAX_TIME_OFFSET`, node/GPU counts, batch sizing). The active rows train the
"huge" and "giant" configurations across 10 nodes; smaller/commented rows are
retained for reference.

Launch
------

From within this directory, with the Yoke environment active:

```bash
yoke-start-study --csv ddp_paper_study.csv --submissionType slurm
```

Add `--dryrun` to render the study directories and print the `sbatch` commands
without submitting any jobs.
