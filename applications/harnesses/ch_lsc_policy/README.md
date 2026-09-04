Layered Shaped Charge Policy Training - DDP - Chicoma
=====================================================

Chicoma (`ch_`) DDP harness for pre-training a Gaussian **policy network** used in
the reinforcement-learning design of layered shaped charge (LSC) geometries.

Files
-----

- `train_lsc_policy.py` — DDP training script for the policy network.
- `training_input.tmpl` — single input template (with the
  `# <<optional:CONTINUATION>>` block for continuation runs).
- `training_slurm.tmpl` — complete SLURM submission script.
- `cp_files.txt` — files copied into each `study_###` run directory.
- `lsc_policy_params.csv` — a single active study (row `11`).

Study
-----

`lsc_policy_params.csv` exposes the cosine LR schedule (`ANCHOR_LR`, `NUM_CYCLES`,
`MIN_FRACTION`, `TERMINAL_STEPS`, `WARMUP_STEPS`), batch size, and per-epoch batch
counts. The commented rows record prior LR-tuning sweeps and are retained for
reference.

Launch
------

From within this directory, with the Yoke environment active:

```bash
yoke-start-study --csv lsc_policy_params.csv --submissionType slurm
```

Add `--dryrun` to render the study directories and print the `sbatch` commands
without submitting any jobs.
