LodeRunner Training - Lightning - Chicoma
=========================================

Chicoma (`ch_`) harness that trains a `lightning`-wrapped LodeRunner on
`lsc240420` per-material density and velocity fields. `lightning` provides the
multi-node, multi-GPU distributed-data-parallel training. A sequence of
timesteps is input and predicted, with scheduled sampling over the sequence.

Files
-----

- `train_LodeRunner_lightning.py` — LightningModule training script.
- `training_input.tmpl` — single input template (with the
  `# <<optional:CONTINUATION>>` block for continuation runs).
- `training_slurm.tmpl` — complete SLURM submission script.
- `cp_files.txt` — files copied into each `study_###` run directory.
- `hyperparameters.csv` — a single active study (row `123`).

Study
-----

`hyperparameters.csv` configures the LodeRunner model shape (`B0`–`B3`), the
cosine LR schedule (`ANCHOR_LR`, `NUM_CYCLES`, `MIN_FRACTION`, `TERMINAL_STEPS`,
`WARMUP_STEPS`), the input sequence length (`SEQ_LEN`), and the scheduled-sampling
schedule (`SCHEDULE`, `INITIAL_SCHEDULE_PROB`, `DECAY_PARAM`,
`MINIMUM_SCHEDULE_PROB`). The commented row is retained for reference.

Launch
------

From within this directory, with the Yoke environment active:

```bash
yoke-start-study --csv hyperparameters.csv --submissionType slurm
```

Add `--dryrun` to render the study directories and print the `sbatch` commands
without submitting any jobs.
