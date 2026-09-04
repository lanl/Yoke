LSC Action Network Training
===========================

Trains a Transpose-CNN (TCNN) surrogate — the **action network** for the layered
shaped charge (LSC) reinforcement-learning design pipeline. The surrogate maps LSC
simulation geometry parameters to density images and is used as the action network
in the RL loop for LSC design.

Files
-----

- `train_lsc_action.py` — main training script: model initialization, dataset
  preparation, training loop, LR scheduling, checkpointing, and SLURM integration.
- `training_input.tmpl` — single input template (with the
  `# <<optional:CONTINUATION>>` block for continuation runs).
- `training_slurm.tmpl` — complete SLURM submission script.
- `cp_files.txt` — files copied into each `study_###` run directory.
- `hyperparameters.csv` — a single active study (row `1`).

Study
-----

`hyperparameters.csv` configures the TCNN feature widths (`LINEAR_F`, `F0`–`F4`),
the base learning rate (`LEARN_RATE`), batch sizing, and per-epoch batch counts.
Training integrates a cosine learning-rate scheduler with warmup
(`CosineWithWarmupScheduler`) for dynamic LR adaptation.

Launch
------

From within this directory, with the Yoke environment active:

```bash
yoke-start-study --csv hyperparameters.csv --submissionType slurm
```

Add `--dryrun` to render the study directories and print the `sbatch` commands
without submitting any jobs.
