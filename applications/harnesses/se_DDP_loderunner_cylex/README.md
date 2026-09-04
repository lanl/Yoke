LodeRunner Training - DDP - Selene - CYLEX
==========================================

Selene (`se_`) DDP LodeRunner harness trained on the `cx241203` (CYLEX)
experimental/simulation dataset instead of `lsc240420`. Uses PyTorch
`DistributedDataParallel` on single-timestep input/output of the per-material
density fields, with a cosine learning-rate schedule with warmup
(`CosineWithWarmupScheduler`).

Files
-----

- `train_LodeRunner_ddp_cylex.py` — DDP training script (CYLEX dataset).
- `training_input.tmpl` — single input template (with the
  `# <<optional:CONTINUATION>>` block for continuation runs).
- `training_slurm.tmpl` — complete SLURM submission script.
- `cp_files.txt` — files copied into each `study_###` run directory.
- `ddp_production.csv` — longer-epoch production run after tuning.
- `ddp_test.csv` — short smoke-test configuration.

Study
-----

Both CSVs configure a single study each. Beyond the LodeRunner model shape
(`EMBED_DIM`, `B0`–`B3`) and data-loading knobs, the CYLEX runs expose the cosine
LR schedule parameters: `ANCHOR_LR`, `NUM_CYCLES`, `MIN_FRACTION`,
`TERMINAL_STEPS`, and `WARMUP_STEPS`. `ddp_test.csv` is the quick verification
run; `ddp_production.csv` is the full production configuration.

Launch
------

From within this directory, with the Yoke environment active:

```bash
# quick smoke test
yoke-start-study --csv ddp_test.csv --submissionType slurm

# production run
yoke-start-study --csv ddp_production.csv --submissionType slurm
```

Add `--dryrun` to render the study directories and print the `sbatch` commands
without submitting any jobs.
