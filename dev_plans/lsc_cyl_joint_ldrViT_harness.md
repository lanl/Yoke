# Joint LSC + CYL LodeRunnerViT (2-frame) Harness

## Goal

Train one 2-frame `LodeRunnerViT` on LSC (`lsc240420`) and Cylex (`cx241203`)
PAGOSA data together, or on either dataset alone, from a single harness.

Both datasets come from the same physics code, but they name fields
differently. The plan adds:

- a **canonical, material-based channel vocabulary**, used as the model's
  `default_vars`;
- datasets that translate their own field names into canonical channel indices
  and return a **per-sample channel map**;
- lead times computed from each frame's **`sim_time`**;
- a **dataset tag** attached to every per-sample loss record.

## Amendment 1 (2026-10-30): train with the per-channel LpLoss, not MSE

The first implementation followed `ch_ldrViT/train_ldrViT_2frame.py` and so
trained with `HarnessTrainer`'s default `nn.MSELoss(reduction="none")`
(`src/yoke/utils/builders.py:122`). MSE gives poor training results,
especially for autoregressive rollouts, while the per-channel L2-norm loss
largely fixes this even without normalization. The harness is therefore
re-based on `ch_ldrViT/train_ldrViT_2frame_8ch_lp.py`.

That script differs from `train_ldrViT_2frame.py` in exactly three ways. The
joint harness takes on the first two (loss and datasteps) and deliberately
**not** the third (channel set):

1. **Loss**: `loss_builder=build_lp_loss`, i.e.
   `LpLoss(d=2, p=2, method="abs", eps=1e-4, reduction="none")`, which
   returns a `(B, C)` per-channel L2 norm over `(H, W)`.
2. **Datasteps**: the `pli_2frame_lp` steps
   (`datastep/loderunner.py:437-579`) backpropagate
   `per_channel_loss.mean()` (mean over batch *and* channel) and record and
   gather `per_channel_loss.mean(dim=1)` (one value per sample).
3. **Channels**: an 8-channel target, the 6 material densities plus the 2
   velocities, with no energy or pressure. **Not adopted**: per the user, the
   joint harness keeps training on density, energy *and* pressure for each
   material (closer to `train_ldrViT_2frame_20ch_lp_norm.py`'s channel set,
   without the normalization).

Changes (see the updated Design, Steps, and Verification sections):

- **Varchan datasteps** (`train_/eval_DDP_loderunner_2frame_varchan_datastep`)
  become LpLoss-only, using the same reduction as the `pli_2frame_lp` steps.
  `loss_fn` must return `(B, C)`. Any other shape (for example an elementwise
  MSE's `(B, C, H, W)`) raises a `ValueError`, so a mis-wired loss fails at the
  first step instead of training silently on the wrong objective. Their
  docstrings are updated to say so.
- **Harness script** gains `build_lp_loss()` (copied from
  `train_ldrViT_2frame_8ch_lp.py`) and `loss_builder=build_lp_loss`. Its
  module docstring and `descr_str`/`prog` are updated to mention the LpLoss.
- **Channel set unchanged**: `training_input.tmpl` keeps
  `--thermo_kinds density energy pressure`. On LSC that is 20 channels
  (6 materials × 3 kinds + 2 velocities); on CYL it is the 2 velocities plus
  3 kinds per material present. The script's `--thermo_kinds` default changes
  from `["density"]` to all three, so the default matches the intended study.
  The LpLoss is per channel, so channels of very different scale (densities
  vs. pressures/energies) are balanced without normalization.
- **Loss-record scale**: records now hold the per-channel L2 norm averaged over
  channels, the same quantity `8ch_lp` records. This is much larger than an
  MSE value because the norm has no 1/M averaging. The README notes that
  `yoke-plot-loss` needs a larger `--ylim` (its default is 1.0).
- **Tests**:
  - The varchan datastep tests switch to a real `LpLoss`. They check the
    recorded per-sample value against `LpLoss(...).mean(dim=1)`, check that
    the backward pass reduces with `mean()` (gradient equal to that of
    `per_channel_loss.mean()`), and check that an elementwise loss raises.
  - The multidataset epoch tests are unchanged, since they mock the
    datasteps.
- **Not changed**:
  - `train_DDP_loderunner_multidataset_epoch` (the loss is just passed
    through).
  - The datasets.
  - `MixedDataSet`.
  - The plotting code.
  - The existing `pli_2frame_lp` / `pli_2frame` datasteps.
  - `HarnessTrainer`.
  - `DATASTEP_FN` (the varchan steps are called directly by the epoch
    function, not through the table).

## What the sample data shows (`data_subsamples/`)

Verified by loading the files directly:

- **LSC** `lsc240420_id01124_pvi_idx0000{1,2,3}.npz` has 66 keys. Fields use
  part names: `{density,energy,pressure}_{case,cushion,maincharge,outside_air,
  striker,throw}`, plus `vofm_<part>`, `Uvelocity`, `Wvelocity`, `Rcoord`
  (400,), `Zcoord` (1120,), and `sim_time` (a 0-d float64 scalar). Material
  fields are NaN outside the material.
- **CYL** `cx241203_id00989_pvi_idx0000{0,2,3}.npz` has 35 keys. Index 00001
  is missing, which is a real gap the sampler must tolerate. Fields are a
  *mix* of conventions:
  - Part-named: `*_wall`, `*_maincharge`, `*_booster`, each with its own
    `vofm_*`.
  - Material-named: the back material only (here `*_Ta`, `vofm_Ta`).
  - There is no air field in this simulation. `vofm_wall + vofm_Ta +
    vofm_booster + vofm_maincharge` equals 1 everywhere, so the back material
    fills the outside region.
  - `sim_time` is present (a 0-d scalar). Images are 1120×400 half-images.
- **`sim_time` vs. `0.25·idx`.** LSC: 0.2540, 0.5031, 0.7509. CYL: 0.0,
  0.5029, 0.7504. The values are close to, but not exactly, 0.25·idx.
- **Booster vs. maincharge.** In the CYL sample their volume fractions never
  overlap (0 shared cells).
- **Grids.** Both use 1120×400 cells at the same 0.025 spacing (Rcoord 0–10
  in both). However, **Zcoord spans differ: LSC is −4 to 24, CYL is −8 to 20.**
- **Design CSV** `design_cx241203_MASTER.csv` has 2660 rows, indexed by
  `key`. Header names and values have leading spaces (for example
  `' Polymer.Sylgard'`), so both must be stripped. `wallMat`/`backMat` values
  are `Air, Al, Be, Cu, N, Polymer.Sylgard, Sn, Steel.alloySS304L, Ta, U.DU,
  Water, Void`. The 20 `wallMat=Void` rows all have `backMat=Void`.
  `wallMat == backMat` occurs in 220 non-Void rows. For id00989:
  `wallMat=Polymer.Sylgard`, `backMat=Ta`.

## Decisions (confirmed with user)

- Part to material map (canonical names are the CYL material names):

  | LSC part      | CYL part | canonical material  |
  | ------------- | -------- | ------------------- |
  | `case`        |          | `Steel.alloySS304L` |
  | `cushion`     |          | `Polymer.Sylgard`   |
  | `outside_air` |          | `Air`               |
  | `striker`     |          | `Al`                |
  | `throw`       |          | `Cu`                |
  | `maincharge`  | `maincharge`, `booster` | `HE`        |
  |               | `wall`   | design CSV `wallMat` (per simulation) |
  |               | back material | already named by material (`backMat`) |

- **Per-sample channel map, batch size 1** for now. A batch with more than one
  sample is accepted only if every sample has an identical map; otherwise the
  step raises an error.
- **2-frame** setup, mirroring `ch_ldrViT/train_ldrViT_2frame.py` (EMA,
  grad clip, cosine-with-warmup LR, RoPE ViT).
- **Lead times come from `sim_time`**: `dt_in = t(curr) − t(prev)` and
  `dt_out = t(next) − t(curr)`. Frame indices are still sampled as two gaps in
  `[1, max_timeIDX_offset]`. A sample whose dt is ≤ 0, or which has no
  `sim_time`, is treated as a bad sample and resampled.
- **Dataset tag in the loss records**, via a new epoch function (below).
- Normalization is **deferred**. Everything is keyed on canonical names so it
  can be added later.

## Defaults I chose (flag any you disagree with)

1. **Canonical vocabulary** (append-only, because a channel's index is its
   identity in checkpoints): `Uvelocity, Wvelocity`, then for each material in
   `Air, Al, Be, Cu, U.DU, HE, N, Sn, Steel.alloySS304L,
   Polymer.Sylgard, Ta, Water`, the fields `density_`, `energy_`, `pressure_`.
   That is 38 channels. **Void is excluded** (it has no mass or state). The
   HE material is named **`HE`**; the `maincharge` and `booster` *parts* both
   map to it (they may start in different initial states, which the per-cell
   merge handles). `Rcoord`/`Zcoord` are dropped, since
   the ViT uses RoPE.
2. **Merging two sources into one canonical material.** This happens for
   CYL booster+maincharge, and for `wall`+back when `wallMat == backMat`.
   Densities are first multiplied by volume fraction, as the existing loaders
   do (`density_X * vofm_X`). Then:
   - **density**: sum the volume-fraction-weighted densities. This is exact.
   - **pressure**: volume-fraction-weighted mean
     `Σ p_i·vf_i / Σ vf_i`, with 0 where `Σ vf_i = 0`.
   - **energy** (specific): mass-weighted mean
     `Σ e_i·ρ_i·vf_i / Σ ρ_i·vf_i`, with 0 where the mass is 0.

   With no overlap (booster/maincharge in the sample) these all reduce to the
   simple fill used by `combine_by_number_and_label`. I chose these instead of
   that fill because a simulation with `wallMat == backMat` can have mixed
   cells.

   *Not verified*: whether NPZ files with `wallMat == backMat` contain both
   `*_wall` and `*_<mat>` fields. The code handles either case.
3. **Void handling.** Parts that resolve to `Void` are skipped. A requested
   field missing from any of the three frames is dropped from that sample.
   Missing frames, such as CYL idx00001, are resampled, matching the
   existing retry behavior.
4. **Stable dataset IDs.** `DATASET_IDS = {"lsc": 0, "cyl": 1}` lives in the
   field-mapping module. Each dataset returns its own ID, so tags stay
   consistent across runs and continuations, even when one dataset's weight
   is 0.
5. **New module** `src/yoke/datasets/pagosa_2frame.py` holds a shared base
   class and the LSC and CYL subclasses. AGENTS.md prefers editing existing
   files, but here the triplet sampling, `sim_time` lead times, and channel
   resolution are shared by both datasets, and the two legacy modules can stay
   untouched. If you'd prefer, I can put each subclass in `lsc_dataset.py` /
   `load_npz_dataset.py` instead.
6. Harness directory: `applications/harnesses/ch_ldrViT_lsc_cyl/`.

## Design

### `src/yoke/datasets/pagosa_fields.py` (new)
- `CANONICAL_MATERIALS`, `CANONICAL_FIELDS`, and `canonical_index(name)`.
- `LSC_PART_TO_MATERIAL`, and `CYL_FIXED_PART_TO_MATERIAL`
  (`maincharge`/`booster` → `HE`).
- `DATASET_IDS` / `DATASET_NAMES`.
- `resolve_sources(thermo_kinds, kinematic, part_map, available_keys)`
  returns `{canonical_name: [npz_part_field, ...]}` in vocabulary order. It
  keeps only fields present in `available_keys` and skips `Void`.

### `src/yoke/datasets/pagosa_2frame.py` (new)
- `PAGOSA_2frame_DataSet` (base class):
  - Samples three frames (prev/curr/next) with existence checks and retries,
    using the same logic as `LSC_rho2rho_temporal_2frame_DataSet`.
  - Loads each frame: reads `sim_time`, reads the needed fields with
    NaN → 0, applies volume-fraction weighting, then merges sources per
    canonical channel.
  - Reflects images unless `half_image`.
  - Returns `(input_frames (2,C,H,W), target (C,H,W), lead_times (2,),
    channel_map (C,) long, dataset_id () long)`.
  - Subclasses supply the file path and the per-simulation part map.
- `LSC_2frame_canonical_DataSet` uses a fixed part map.
- `CYL_2frame_canonical_DataSet` reads the design CSV **once** in `__init__`,
  stripping header names and values, and keeps only `wallMat`/`backMat`. Its
  part map is `wall → wallMat` plus `backMat → backMat`.
- Kwargs: `thermo_kinds` (default `("density",)`, any subset of
  `density/energy/pressure`) and `kinematic` (default
  `("Uvelocity", "Wvelocity")`).

### `src/yoke/datasets/mixed_dataset.py` (new)
- `MixedDataSet(datasets, weights, length)` splits the index range into
  contiguous blocks in proportion to `weights`. `DistributedSampler`
  shuffling then mixes the datasets randomly, and the split is identical on
  every rank and worker. Datasets with weight 0 are not passed in at all.

### Datasteps (`src/yoke/utils/training/datastep/loderunner.py`, additive)
- `train_DDP_loderunner_2frame_varchan_datastep` and
  `eval_DDP_loderunner_2frame_varchan_datastep`.
- They unpack the 5-tuple and check that all rows of the batched
  `channel_map` match; row 0 is used as both `in_vars` and `out_vars`.
- Forward pass, grad clip, and optimizer step are as in the existing 2-frame
  steps. **Loss (Amendment 1):** LpLoss-only, using the same reduction as the
  `pli_2frame_lp` steps. `loss_fn` returns `(B, C)`; the backward pass uses
  `.mean()` and the record uses `.mean(dim=1)`. Any other shape raises a
  `ValueError`.
- They `all_gather` both the per-sample losses and the per-sample
  `dataset_id`s, and return `(end_img, pred_img, all_losses, all_ids)`.

### Epoch function (`src/yoke/utils/training/epoch/loderunner.py`, additive)
- `train_DDP_loderunner_multidataset_epoch` has the same signature and
  behavior as `train_DDP_loderunner_epoch`: LR step, EMA update, `global_step`
  return, and validation every `train_per_val` epochs.
- It calls the varchan steps directly. No `DATASTEP_FN` entry is needed, and
  the `channel_map` kwarg is not required.
- It writes records as `epoch, batch, loss, dataset`, for example
  `12, 340, 0.00123400, cyl`, using `DATASET_NAMES`.
- Existing epoch functions and their 3-column records are unchanged.

### Loss-curve reader (`src/yoke/plots/loss_curves.py`, small edit)
- `load_record_csv` currently forces 3 column names, so a 4-column file would
  be misparsed. It will accept 3 or 4 columns, giving a `Dataset` column when
  present.
- `yoke-plot-loss` (`cli/plot_loss.py`) gets an opt-in `--dataset <tag>`
  filter that plots only that dataset's records (one figure per dataset).
  *Changed during implementation* from the original `--by_dataset` overlay:
  a filter reuses the existing plotting code unchanged.
- The default output for existing 3-column records is unchanged.

### Harness `applications/harnesses/ch_ldrViT_lsc_cyl/` (new)
- `train_ldrViT_2frame_lsc_cyl.py` is a thin `HarnessTrainer` wrapper based on
  `train_ldrViT_2frame_8ch_lp.py` (Amendment 1), with
  `default_vars=CANONICAL_FIELDS`,
  `epoch_fn=train_DDP_loderunner_multidataset_epoch`,
  `loss_builder=build_lp_loss`, and an optional EMA hook.
  `training_input.tmpl` keeps `--thermo_kinds density energy pressure`, and
  that becomes the script's default too.
- Arguments: the standard `cli` groups, plus `--LSC_NPZ_DIR`, `--NPZ_DIR` and
  `--CSV_FILEPATH` for CYL (all existing `cli` args), and harness-local
  `--lsc_{train,validation}_filelist`, `--cyl_{train,validation}_filelist`,
  `--lsc_weight`, `--cyl_weight`, `--thermo_kinds` (nargs), plus the EMA and
  grad-clip args copied from `train_ldrViT_2frame.py`.
- `training_input.tmpl`, `training_slurm.tmpl`, `cp_files.txt`, a smoke-test
  CSV (batch size 1), and `README.md`.
- CYL filelists are not in `applications/filelists/`. The existing cylex
  harness points to `cx241203_prefixes_*_noBe_noVoid_truncated.txt`, so the
  template will use those names and you supply the files.

## Steps

1. `pagosa_fields.py` and `tests/datasets/test_pagosa_fields.py`: pin the
   vocabulary order, LSC map, booster fold, Void skip, and missing-field skip.
2. `pagosa_2frame.py` and `tests/datasets/test_pagosa_2frame.py`:
   - Synthetic NPZ files in `tmp_path` covering: `sim_time` lead times, a
     missing middle index, booster/maincharge merge, `wallMat == backMat`
     merge (all three physics rules), CSV whitespace stripping, Void sims, and
     the shape of the returned tuple.
   - Unit tests use **only synthetic data** built in `tmp_path`. The
     `data_subsamples/` files are not available in CI, so they are used only
     in the manual verification below, never in `tests/`.
3. `mixed_dataset.py` and its tests: block proportions, routing, and
   validation of `length`/`weights`.
4. Varchan datasteps and their tests, with mocked `dist` as in the existing
   datastep tests. Include the error raised for mismatched maps.
5. The multidataset epoch function and its test, which checks the 4-column
   records, the tags, EMA/`global_step`, and the validation cadence.
6. CPU test: a tiny 2-frame `LodeRunnerViT` runs forward and backward with
   two different `in_vars` sets, and every parameter receives a gradient.
   This guards against DDP unused-parameter errors.
7. `load_record_csv` 3/4-column support and `--dataset`, with tests.
8. Harness directory.
9. Verify (below).
10. **Amendment 1 (LpLoss):** make the varchan datasteps LpLoss-only and update
    their tests; add `build_lp_loss` and `loss_builder` to the harness and
    update its docstrings; change the `--thermo_kinds` default to all three
    kinds (the template already passes all three); update the README (loss
    description, ylim note); re-run verification.

## Verification

- `./.venv/bin/python -m pytest -Werror` (full suite).
- `./.venv/bin/ruff check` and `ruff format --check` on the touched files.
- Manual (not committed) end-to-end data check: build the harness's
  datasets against `data_subsamples/` (prefix files written to a temp dir),
  draw samples from both through `MixedDataSet` and a `DataLoader(batch_size=1)`, and print the
  channel names and lead times.
- `yoke-start-study --dryrun` in the harness directory, to check that the
  templates render.
- Amendment 1: re-run the manual `data_subsamples/` check with the harness's
  `build_lp_loss()`. Run a forward pass, the loss, and a backward pass of a
  tiny model on one real LSC sample and one real CYL sample. Check that the
  loss is `(1, C)` and finite, and that every parameter receives a gradient.
- **Not verifiable locally:** multi-GPU DDP, real training dynamics,
  throughput, and real `wallMat == backMat` NPZ files. Those need a cluster
  smoke run, which you would launch.

## Resolved questions

1. **Zcoord offset** (LSC z ∈ [−4, 24], CYL z ∈ [−8, 20]): ignored. The
   simulations are translation-invariant in z. A spatial-coordinate embedding
   in `LodeRunnerViT` may come later.
2. **Mixed-cell merge**: use vf-weighted pressure and mass-weighted energy
   for now (default 2). The user will check with the PAGOSA developers what
   the code itself does. Keep the merge in one small, well-documented
   function so it is easy to change.
3. **Module placement**: `src/yoke/datasets/pagosa_2frame.py` approved.

## Pre-existing issues noticed (not fixing)

- `LabeledData.cylex_data_loader`: with `thermodynamic_variables="all"`, energy
  is never added, because the energy branch is an `elif`
  (`load_npz_dataset.py:416-438`).
- `train_/eval_DDP_loderunner_datastep_cylex` print debug lines on every step.
- Dataset RNGs are created in `__init__`, so forked DataLoader workers within a
  rank likely draw identical time-index sequences. This is inferred from the
  code, not tested. The new datasets will draw from a per-worker
  `np.random.default_rng()` created lazily in `__getitem__`, which avoids the
  problem.
