LodeRunner-ViT Training - DDP - Selene
======================================

Trains the `LodeRunnerViT` architecture on the `lsc240420` layered-shaped-charge
dataset using PyTorch `DistributedDataParallel` (DDP). This harness targets the
Selene cluster. A single timestep of the per-material density and velocity fields
is input and a single (offset) timestep is predicted.

Files
-----

- `train_ldrViT_ddp.py` — DDP training script for `LodeRunnerViT`.
- `training_input.tmpl` — single input template. The `<KEY>` tokens are filled per
  study row; the `# <<optional:CONTINUATION>>` block adds `--continuation` and
  `--checkpoint` only on epoch continuation.
- `training_slurm.tmpl` — complete SLURM submission script.
- `cp_files.txt` — files copied into each `study_###` run directory.
- `ldrViT_lrsched_260817.csv` — learning-rate sweep for the baseline ViT model.

Study
-----

`ldrViT_lrsched_260817.csv` fixes the `LodeRunnerViT` architecture
(`VIT_EMBED_DIM=512`, `VIT_NUM_LAYERS=12`, `VIT_NUM_HEADS=8`) and sweeps the anchor
learning rate (`ANCHOR_LR`) from `1.0e-4` to `1.0e-3` under a cosine-with-warmup
schedule (`NUM_CYCLES`, `MIN_FRACTION`, `TERMINAL_STEPS`, `WARMUP_STEPS`). The
`allocation` column records the intended account for each row. The leading
commented block lists candidate Selene accounts/partitions for reference.

Launch
------

From within this directory, with the Yoke environment active:

```bash
yoke-start-study --csv ldrViT_lrsched_260817.csv --submissionType slurm
```

Add `--dryrun` to render the study directories and print the `sbatch` commands
without submitting any jobs.
