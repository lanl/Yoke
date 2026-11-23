LodeRunner-ViT Joint LSC + CYL Training - DDP - Chicoma
=======================================================

Trains one 2-frame `LodeRunnerViT` on the `lsc240420` (layered shaped charge)
and `cx241203` (Cylex) PAGOSA datasets at the same time, or on either alone.

The two datasets name their fields differently: LSC uses part names
(`density_throw`), while CYL mixes part names (`wall`, `maincharge`, `booster`)
and material names (the back material). `yoke.datasets.pagosa_fields` maps both
onto one material-based vocabulary, `CANONICAL_FIELDS`, which is the model's
`default_vars`:

| LSC part      | CYL part                  | material            |
| ------------- | ------------------------- | ------------------- |
| `case`        |                           | `Steel.alloySS304L` |
| `cushion`     |                           | `Polymer.Sylgard`   |
| `outside_air` |                           | `Air`               |
| `striker`     |                           | `Al`                |
| `throw`       |                           | `Cu`                |
| `maincharge`  | `maincharge`, `booster`   | `HE`                |
|               | `wall`                    | design CSV `wallMat` |
|               | back material             | design CSV `backMat` |

Training uses the per-channel `LpLoss` (`d=2, p=2, method="abs",
eps=1e-4`) and the LpLoss datastep convention of
`ch_ldrViT/train_ldrViT_2frame_8ch_lp.py`: backprop on the mean over batch and
channel, record the mean over channel. Density, energy and pressure are trained
for every material present (`--thermo_kinds`), which is 20 channels on LSC.

Each sample carries its own channel map (CYL material sets vary between
simulations), so **the batch size is fixed at 1**. Lead times come from each
frame's `sim_time`. When two parts share a material (booster + maincharge, or
`wallMat == backMat`), densities are summed after volume-fraction weighting,
pressure is volume-fraction averaged, and specific energy is mass averaged
(`yoke.datasets.pagosa_2frame.merge_material_parts`).

Files
-----

- `train_ldrViT_2frame_lsc_cyl.py` — DDP training script.
- `training_input.tmpl` — input template, with a `# <<optional:CONTINUATION>>`
  block for continuation runs.
- `training_slurm.tmpl` — SLURM submission script (copied from `ch_ldrViT`).
- `cp_files.txt` — files copied into each `study_###` directory.
- `ldrViT_2frame_lsc_cyl_smoke.csv` — smoke tests: joint (study 1), LSC only
  (study 2), and CYL only (study 3).

Study
-----

`LSC_WEIGHT` and `CYL_WEIGHT` set the relative sampling probability of each
dataset; a weight of `0` skips that dataset entirely. The CYL prefix lists named
in `training_input.tmpl` (`cx241203_prefixes_*_noBe_noVoid_truncated.txt`) are
not in `applications/filelists/` and must be provided.

Training and validation records have a fourth column naming the source dataset
(`epoch, batch, loss, dataset`). Plot one dataset's curves with, for example:

```bash
yoke-plot-loss --basedir ./runs --IDX 1 --dataset cyl --ylim 50
```

The recorded values are L2 norms (no averaging over pixels), so they are much
larger than MSE values; raise `--ylim` (default `1.0`) to fit them.

Launch
------

From within this directory, with the Yoke environment active:

```bash
yoke-start-study --csv ldrViT_2frame_lsc_cyl_smoke.csv --submissionType slurm
```

Add `--dryrun` to render the study directories and print the `sbatch` commands
without submitting any jobs.
