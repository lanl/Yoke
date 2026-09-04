LSC Inverse (Parameter-Estimation) Training - DDP - Chicoma
===========================================================

Chicoma (`ch_`) DDP harness that trains a parameter-estimation CNN for the
layered shaped charge (LSC) problem — an "inverse" model that maps density-field
observations back to the geometry parameters that produced them.

Files
-----

- `train_lsc_inverse.py` — DDP training script for the parameter-estimation CNN.
- `training_input.tmpl` — single input template (with the
  `# <<optional:CONTINUATION>>` block for continuation runs).
- `training_slurm.tmpl` — complete SLURM submission script.
- `cp_files.txt` — files copied into each `study_###` run directory.
- `lsc_inverse_params.csv` — a single active study (row `11`).

Study
-----

`lsc_inverse_params.csv` exposes the cosine LR schedule (`ANCHOR_LR`,
`NUM_CYCLES`, `MIN_FRACTION`, `TERMINAL_STEPS`, `WARMUP_STEPS`) plus batch sizing
and per-epoch batch counts (`NTRN_BATCH`, `NVAL_BATCH`). The commented rows record
prior gradient-flow and LR-tuning sweeps and are retained for reference.

Launch
------

From within this directory, with the Yoke environment active:

```bash
yoke-start-study --csv lsc_inverse_params.csv --submissionType slurm
```

Add `--dryrun` to render the study directories and print the `sbatch` commands
without submitting any jobs.
