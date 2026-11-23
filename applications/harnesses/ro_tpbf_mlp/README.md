TPBF melt-pool MLP - k-fold - rocinante (CPU)
=============================================

Rocinante (`ro_`) harness that trains a small SwiGLU MLP to predict melt-pool
size (`depth_um`, `width_um`, `fwhm_um`, `length_um`) from TPBF process and
material parameters, one row per Truchas simulation. Each study is one split of
a **5-fold cross-validation repeated with two shuffles** (10 studies), to judge
whether about 100 simulations are enough data for this problem.

Rocinante is CPU-only, so the train script runs `HarnessTrainer` in CPU mode
(`device_type="cpu"`): one process on one node per study, no GPUs, no NCCL.

Data and model
--------------

- Table: `tpbf_training_set_261105.csv` (105 rows), copied into this
  directory and, through `cp_files.txt`, into every study directory.
- Rows: only `melted == True` (98 rows). Non-plateaued rows are kept.
- Inputs: `power_W`, `scan_speed_mps`, `sigma_m`, `S_wpct` (`log10`).
  `O_wpct` is left out because it is zero in every row of this table.
  `T0_K` (the Marangoni flow-reversal temperature) is a simulation output and
  is never an input.
- Folds: stratified by `stratum` (`base` / `transition`), set by
  `--fold` and `--split_seed`. Train folds are 78-79 rows; held-out folds are
  19-20.
- Normalization: inputs and targets are z-scored with the **training fold's**
  statistics (`yoke.datasets.tabular_csv.TabularCSV_DataSet`).
- Model: `yoke.models.mlp.SwiGLU_mlp`, 4 -> [64, 64, 64] -> 4, pre-LayerNorm
  residual SwiGLU layers, 17,804 parameters.
- Loss: `yoke.losses.lp_loss.VectorLpLoss`, the per-sample L2 norm of the
  z-scored error vector. Recorded losses are in z units (not squared).
- Optimizer: AdamW (weight decay 0.01). LR: cosine with warmup, 1e-3 down to
  1e-5 over the run, stepped every batch.

Files
-----

- `train_tpbf_mlp.py`: CPU-mode `HarnessTrainer` wrapper.
- `training_input.tmpl`: argument file template (ends with the
  `# <<optional:CONTINUATION>>` block).
- `training_slurm.tmpl`: one node, one task, `OMP_NUM_THREADS=4`.
  Partition, QOS, and wall time come from the CSV.
- `cp_files.txt`: the train script and the data table.
- `tpbf_kfold.csv`: 10 production studies (folds 0-4 x seeds 0, 1) on
  `standard`, 1000 epochs in one job each (one cycle: multi-cycle runs restart
  the LR warmup, see `dev_plans/lr_scheduler_continuation_bug.md`).
- `tpbf_kfold_smoke.csv`: studies 901-902 on `debug`, 4 epochs in 2 cycles
  (one continuation each).
- `tpbf_kfold_reg.csv`: studies 21-30, a regularized MLP on the same splits:
  [32, 32], weight decay 0.1, 150 epochs in one job.
- `tpbf_kfold_fix.csv`: studies 31-40, the same settings as 21-30, with the
  fixed model (no LayerNorm on the input layer; studies 1-30 used the old
  model, which discarded each input row's mean and scale).
- `summarize_kfold.py`: after training, scores every held-out row from each
  study's latest checkpoint, in physical units.
- `baselines_kfold.py`: training-mean, linear, and quadratic least-squares
  baselines on the same splits and normalization, with the same metrics.
  It needs only the rendered study input files, not checkpoints.

Study columns
-------------

| column | meaning |
| --- | --- |
| `FOLD`, `SPLIT_SEED` | which k-fold split the study trains on |
| `INPUT_COLS` | colon-separated input columns (sweep input sets here) |
| `TOTAL_EPOCHS`, `CYCLE_EPOCHS` | epoch budget, and epochs per SLURM job |
| `TERMINAL_STEPS` | LR schedule length; 19 x `TOTAL_EPOCHS` |
| `TRAIN_PER_VAL` | validate every N epochs |
| `HIDDEN_DIMS` | colon-separated hidden widths, e.g. `64:64:64`, `32:32` |
| `WEIGHT_DECAY` | AdamW weight decay |
| `PARTITION`, `QOS`, `WALLTIME` | SLURM settings per job |
| `train_script`, `allocation` | script name; SLURM account (`xd`) |

Launch
------

From this directory, with the Yoke environment active. Smoke test first:

```bash
yoke-start-study --csv tpbf_kfold_smoke.csv --rundir ./runs \
    --cpFile cp_files.txt --submissionType slurm
```

Then, once both smoke studies have resubmitted and finished:

```bash
yoke-start-study --csv tpbf_kfold.csv --rundir ./runs \
    --cpFile cp_files.txt --submissionType slurm
```

Add `--dryrun` to render the study directories and print the `sbatch`
commands without submitting.

Results
-------

```bash
yoke-plot-loss --basedir ./runs --IDX 1      # loss curves of study 1
python summarize_kfold.py --rundir ./runs --csv tpbf_kfold.csv
```

Compare MLP runs with the baselines (the summary CSVs must come from the same
splits, as they do for `tpbf_kfold.csv` and `tpbf_kfold_reg.csv`):

```bash
python summarize_kfold.py --rundir ./runs --csv tpbf_kfold_reg.csv \
    --out_prefix kfold_reg_summary
python baselines_kfold.py --rundir ./runs --csv tpbf_kfold.csv \
    --compare mlp=kfold_summary_metrics.csv \
    --compare mlp_reg=kfold_reg_summary_metrics.csv
```

`summarize_kfold.py` prints the RMSE, MAE, and median relative error per
target for each study, then the mean and std over the 10 studies. It also
writes `kfold_summary_metrics.csv` and `kfold_summary_predictions.csv`.

Caveats
-------

- The held-out fold is used for both validation and test. There is no
  checkpoint selection on it, so the reported numbers come from the final
  checkpoint.
- The validation loader drops the last incomplete batch and does not shuffle.
  So the recorded validation loss leaves out the same 0-3 held-out rows every
  epoch. `summarize_kfold.py` scores all of them.
- Each study holds a whole `standard` node (`OverSubscribe=EXCLUSIVE`) for a
  few minutes while using few threads.
- Later study: add `max_T_K` and `umax_mps` to `INPUT_COLS`. `O_wpct` can be
  added once a table has non-zero values; the dataset rejects a constant
  column.
