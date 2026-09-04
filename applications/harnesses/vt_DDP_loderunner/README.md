LodeRunner Training - DDP - Venadito
====================================

Trains LodeRunner using PyTorch `DistributedDataParallel` (DDP) on the `lsc240420`
layered-shaped-charge dataset, targeting the Venadito cluster. A single timestep
of the per-material density and velocity fields is input and a single (offset)
timestep is predicted.

Files
-----

- `train_LodeRunner_ddp.py` — DDP training script for LodeRunner.
- `training_input.tmpl` — single input template. The `<KEY>` tokens are filled per
  study row; the `# <<optional:CONTINUATION>>` block adds `--continuation` and
  `--checkpoint` only on epoch continuation.
- `training_slurm.tmpl` — complete SLURM submission script.
- `cp_files.txt` — files copied into each `study_###` run directory.
- `ddp_paper_study.csv` — LodeRunner model-size sweep.
- `slurm_acct_limit_inspection.sh` — helper for inspecting SLURM account limits.

Study
-----

`ddp_paper_study.csv` sweeps LodeRunner model size via the embedding dimension
(`EMBED_DIM`, from 96 up to 512) and Swin block structure (`B0`–`B3`) at a fixed
anchor learning rate (`ANCHOR_LR=1e-4`) and max time-index offset
(`MAX_TIME_OFFSET=1`). Node/GPU counts and batch sizing are set per row.

Launch
------

From within this directory, with the Yoke environment active:

```bash
yoke-start-study --csv ddp_paper_study.csv --submissionType slurm
```

Add `--dryrun` to render the study directories and print the `sbatch` commands
without submitting any jobs.
