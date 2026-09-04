LSC Reward Network Training - DDP - Chicoma
===========================================

Chicoma (`ch_`) DDP harness that trains a **reward network** for the layered
shaped charge (LSC) reinforcement-learning design pipeline. The reward network
scores the error between a current density field and a target density field, and
is used as the reward signal when optimizing LSC geometries.

Files
-----

- `train_lsc_reward.py` — DDP training script for the reward network.
- `training_input.tmpl` — single input template (with the
  `# <<optional:CONTINUATION>>` block for continuation runs).
- `training_slurm.tmpl` — complete SLURM submission script.
- `cp_files.txt` — files copied into each `study_###` run directory.
- `hyperparameters.csv` — a single active study (row `1`).

Study
-----

`hyperparameters.csv` exposes the cosine LR schedule (`ANCHOR_LR`, `NUM_CYCLES`,
`MIN_FRACTION`, `TERMINAL_STEPS`, `WARMUP_STEPS`), batch size, and per-epoch batch
counts. The comments note memory limits: `batch_size=16` causes CUDA OOM;
`batch_size=8` with 100 batches/epoch yields ~4-minute epochs.

Launch
------

From within this directory, with the Yoke environment active:

```bash
yoke-start-study --csv hyperparameters.csv --submissionType slurm
```

Add `--dryrun` to render the study directories and print the `sbatch` commands
without submitting any jobs.
