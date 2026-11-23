TPBF melt-pool MLP with in-situ measurables - k-fold - rocinante (CPU)
=====================================================================

Copy of `../ro_tpbf_mlp/` (read its README for the data, model, loss, and
caveats). The only difference: the forward model also takes the in-situ
"measurables" `max_T_K` (maximum melt-pool surface temperature) and
`umax_mps` (maximum surface flow speed) as inputs. The question is whether
measuring them in situ would improve melt-pool size prediction enough to
justify the instrumentation.

- Inputs: `power_W`, `scan_speed_mps`, `sigma_m`, `S_wpct` (log10),
  `max_T_K`, `umax_mps`. Targets: `depth_um`, `width_um`, `fwhm_um`,
  `length_um`.
- `tpbf_meas_kfold.csv`: studies 1-10. They use the same 10 splits and
  settings as `ro_tpbf_mlp` studies 31-40 (`tpbf_kfold_fix.csv`): the fixed
  `SwiGLU_mlp` 6 -> [32, 32] -> 4 (2,756 parameters), weight decay 0.1, and
  150 epochs in one job.
- Job names: `tpbfM_s<NNN>_e<EEEE>`. `runs` is a symlink to
  `/lustre/roscratch1/hickmank/tpbf_meas_261118`.

The splits depend only on the filtered rows and the strata, so study N here
holds out the same simulations as `ro_tpbf_mlp` study 30+N. Compare directly:

```bash
yoke-start-study --csv tpbf_meas_kfold.csv --rundir ./runs \
    --cpFile cp_files.txt --submissionType slurm
python summarize_kfold.py --rundir ./runs --csv tpbf_meas_kfold.csv \
    --out_prefix kfold_meas_summary
python baselines_kfold.py --rundir ./runs --csv tpbf_meas_kfold.csv \
    --compare mlp_meas=kfold_meas_summary_metrics.csv \
    --compare mlp_no_meas=../ro_tpbf_mlp/kfold_fix_summary_metrics.csv
```

The baselines here use the 6 inputs too. The `mlp_no_meas` row is the
4-input MLP on the same splits.
