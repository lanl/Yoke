LodeRunner-ViT Training - DDP - Chicoma
=======================================

Trains the `LodeRunnerViT` architecture on the `lsc240420` layered-shaped-charge
dataset using PyTorch `DistributedDataParallel` (DDP). This harness targets the
Chicoma cluster and supports two input/output framings via two training scripts.

Files
-----

- `train_ldrViT_ddp.py` — DDP training script for `LodeRunnerViT` (single-timestep
  in, single (offset) timestep out).
- `train_ldrViT_2frame.py` — DDP training script for the two-frame framing of
  `LodeRunnerViT`.
- `training_input.tmpl` — single input template. The `<KEY>` tokens are filled per
  study row; the `# <<optional:CONTINUATION>>` block adds `--continuation` and
  `--checkpoint` only on epoch continuation.
- `training_slurm.tmpl` — complete SLURM submission script.
- `cp_files.txt` — files copied into each `study_###` run directory (both train
  scripts, since the CSV's `train_script` column selects which one runs per study).
- `ldrViT_lrsched_260813.csv` — learning-rate-schedule sweep over ViT model sizes.
- `ldrViT_2frame_260903.csv` — two-frame baseline study.

Study
-----

`ldrViT_lrsched_260813.csv` sweeps `LodeRunnerViT` architecture scalings —
embedding dimension (`VIT_EMBED_DIM`), depth (`VIT_NUM_LAYERS`), and attention
heads (`VIT_NUM_HEADS`) — crossed with the anchor learning rate (`ANCHOR_LR`) and
cosine-with-warmup schedule parameters (`NUM_CYCLES`, `MIN_FRACTION`,
`TERMINAL_STEPS`, `WARMUP_STEPS`). The `allocation` column records the intended
account for each row.

`ldrViT_2frame_260903.csv` trains the UMich baseline two-frame configuration.

The `train_script` column selects which training script each row invokes. The
leading commented block lists candidate Chicoma accounts/partitions for reference.

Launch
------

From within this directory, with the Yoke environment active:

```bash
yoke-start-study --csv ldrViT_lrsched_260813.csv --submissionType slurm
```

Add `--dryrun` to render the study directories and print the `sbatch` commands
without submitting any jobs.
