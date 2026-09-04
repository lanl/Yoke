LodeRunner Training - DDP - Chicoma
===================================

An example setup training LodeRunner using PyTorch `DistributedDataParallel`
(DDP) on the `lsc240420` layered-shaped-charge dataset. A single timestep of the
per-material density and velocity fields is input and a single (offset) timestep
is predicted. The training system works within limitations but seems more stable
than `lightning.fabric`.

Files
-----

- `train_LodeRunner_ddp.py` — DDP training script for LodeRunner.
- `training_input.tmpl` — single input template. The `<KEY>` tokens are filled
  per study row; the `# <<optional:CONTINUATION>>` block adds `--continuation`
  and `--checkpoint` only on epoch continuation.
- `training_slurm.tmpl` — complete SLURM submission script (Venado GPU partition).
- `cp_files.txt` — files copied into each `study_###` run directory.
- `ddp_paper_study.csv` — hyperparameters for the LodeRunner-18channel paper runs.

Study
-----

`ddp_paper_study.csv` sweeps LodeRunner model size and temporal offset. Each row
varies the embedding dimension (`EMBED_DIM`), Swin block structure
(`B0`–`B3`), the max time-index offset between input/output (`MAX_TIME_OFFSET`),
node/GPU counts (`KNODES`, `NGPUS`), and batch sizing. The active rows train the
"huge" (`EMBED_DIM=352`) and "giant" (`EMBED_DIM=512`) configurations across 10
nodes; smaller/commented rows are kept for reference.

Platform notes
--------------

1. On Venado, beyond 4 nodes communication conflicts arise intermittently, e.g.:

   ```
   RuntimeError: CUDA error: uncorrectable ECC error encountered
   ```

2. On Venado, with 8 `lsc240420` fields each GPU can only fit 5 samples at a time.
3. On Chicoma, the Giant-size LodeRunner will not fit with DDP training.
4. On Chicoma, the Big-size LodeRunner handles per-GPU batch sizes of 10.

Launch
------

From within this directory, with the Yoke environment active:

```bash
yoke-start-study --csv ddp_paper_study.csv --submissionType slurm
```

Add `--dryrun` to render the study directories and print the `sbatch` commands
without submitting any jobs.
