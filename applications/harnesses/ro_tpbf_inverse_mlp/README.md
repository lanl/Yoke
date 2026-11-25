TPBF inverse MLP - k-fold - rocinante (CPU)
===========================================

The inverse of `../ro_tpbf_mlp/`: from a desired weld depth and observed
melt-pool quantities, predict laser process parameters. The data, splits,
model, loss, and training recipe are the same as `ro_tpbf_mlp` studies 31-40
(read its README). `train_tpbf_inverse_mlp.py` is `train_tpbf_mlp.py` with
inverse defaults; each CSV sets the input and target columns.

| CSV | studies | inputs | targets |
| --- | --- | --- | --- |
| `tpbf_inv_kfold.csv` | 1-10 | `depth_um`, `S_wpct` (log10), `length_um`, `width_um` | `power_W`, `scan_speed_mps`, `sigma_m` |
| `tpbf_inv_meas_kfold.csv` | 11-20 | the above + `max_T_K`, `umax_mps` | `power_W`, `scan_speed_mps`, `sigma_m` |
| `tpbf_inv_sigma_kfold.csv` | 21-30 | the first set + a known `sigma_m` | `power_W`, `scan_speed_mps` |

Every CSV uses the 10 splits of `ro_tpbf_mlp` (folds 0-4 x seeds 0, 1), with
SwiGLU_mlp [32, 32], weight decay 0.1, and 150 epochs in one job. Job names
are `tpbfI_s<NNN>_e<EEEE>`. `runs` is a symlink to
`/lustre/roscratch1/hickmank/tpbf_inv_261118`.

Caveats:

- **Non-uniqueness.** An offline quadratic fit shows that depth, length, and
  width barely constrain `sigma_m`: its error was 0.12 vs 0.13 for
  predicting the mean. Where several parameter sets give the same melt pool,
  an L2-trained network predicts roughly their average, which need not be a
  valid recipe. Small errors on `power_W`/`scan_speed_mps` don't guarantee
  the requested depth either. The next check is forward consistency: run
  the predicted parameters back through a forward model (e.g. the
  `ro_tpbf_mlp` fixed MLP or a quadratic fit), and compare the resulting
  depth/length/width with the requested ones.
- The loss weights the targets equally in z units.

Launch and analysis (from this directory):

```bash
for c in tpbf_inv_kfold tpbf_inv_meas_kfold tpbf_inv_sigma_kfold; do
    yoke-start-study --csv $c.csv --rundir ./runs --cpFile cp_files.txt \
        --submissionType slurm
done
python summarize_kfold.py --rundir ./runs --csv tpbf_inv_kfold.csv \
    --out_prefix kfold_inv_summary
python baselines_kfold.py --rundir ./runs --csv tpbf_inv_kfold.csv \
    --compare mlp_inv=kfold_inv_summary_metrics.csv
```

Repeat the last two commands for the `_meas` and `_sigma` CSVs, with their
own `--out_prefix`.
