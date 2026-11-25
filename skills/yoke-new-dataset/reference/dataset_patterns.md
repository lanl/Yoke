# Dataset Patterns in Yoke

The idioms existing Yoke datasets use. Follow them unless the user has a
reason not to. Best modern example: `src/yoke/datasets/pagosa_2frame.py`.
Older but common: `src/yoke/datasets/lsc_dataset.py`.

## Where things go

- The dataset class goes in `src/yoke/datasets/<module>.py`. Extend an
  existing module or subclass an existing class when the data is a variant of
  one already supported. `LSC_2frame_canonical_DataSet` and
  `CYL_2frame_canonical_DataSet` share `PAGOSA_2frame_DataSet` and override
  only `part_map` and `frame_path`.
- Pure helpers (field readers, merges, channel vocabularies) are module-level
  functions so they can be tested alone (see
  `src/yoke/datasets/pagosa_fields.py`).
- Data paths are **constructor arguments**, never constants in the package.
  The harness passes them from CLI flags (`--NPZ_DIR`, `--FILELIST_DIR`, or
  new harness-specific flags).
- Filelists go in `applications/filelists/`.
- Tests go in `tests/datasets/test_<module>.py`.

## Two sampling styles

### Fixed-index (one item per sample file)

- The filelist holds one **file name** per line (for example
  `lsc240420_id03510_pvi_idx00002.npz`).
- `__len__` returns the number of lines. `__getitem__(index)` loads
  `filelist[index % N]`.
- Use it when each file is one complete sample and epochs should visit each
  once. Example: `yoke.datasets.lsc_dataset.LSC_cntr2rho_DataSet`.

### Random-sampling (draw samples from simulations)

- The filelist holds one simulation **prefix** per line (for example
  `lsc240420_id03366`), and frames are found by pattern
  (`f"{prefix}_pvi_idx{idx:05d}.npz"`).
- `__len__` returns `int(1e6)`. The trainer caps work per epoch with
  `--train_batches`/`--val_batches`, so "length" is effectively infinite.
- `__getitem__(index)` picks prefix `index % N`, then draws time indices
  (gaps up to `max_timeIDX_offset`) with an RNG.
- Use it for time-pair or time-sequence tasks over many simulations.
  Examples: `yoke.datasets.pagosa_2frame.PAGOSA_2frame_DataSet` and
  `yoke.datasets.lsc_dataset.LSC_rho2rho_temporal_DataSet`.

## Filelists

- Read with
  `[line.strip() for line in f if line.strip()]` and raise `ValueError` if
  the list is empty.
- Random-sampling datasets shuffle the prefix list once (`random.shuffle`).
- Train, validation, and test lists are separate files named
  `<dataset>_prefixes_{train_80pct,validation_10pct,test_10pct}.txt` (or
  `<dataset>_{train,val,test}_*.txt` for file lists).
- Splitting by **simulation**, not by frame, avoids leaking near-identical
  frames between train and validation. Ask the user unless they say
  otherwise.
- `applications/makefilelists.py` makes train/validation/test **file** lists
  from a glob (`--input_dir`, `--data_tag`, `--train_split`,
  `--validation_split`, `--test_split`, `--save_dir`). For prefix lists, the
  agent may need a small one-off script. Show it to the user and keep it
  out of `src/yoke`.

## RNG

- Create the generator **lazily** so each `DataLoader` worker gets its own
  rather than a forked copy of one:

  ```python
  self._rng: np.random.Generator | None = None

  @property
  def rng(self) -> np.random.Generator:
      """Per-process random generator, created on first use."""
      if self._rng is None:
          self._rng = np.random.default_rng()
      return self._rng
  ```

- Tests replace `ds._rng` with a fake that returns fixed draws, so keep the
  attribute name `_rng` and draw through `self.rng.integers(...)`.

## Missing or unusable files

Pattern from `PAGOSA_2frame_DataSet.__getitem__`:

1. For up to `max_file_checks` draws on one prefix: build the paths; if any
   file is missing, or `_load_sample` returns `None` (bad time step, missing
   key), draw again.
2. If one prefix is exhausted, print a note to `stderr` and move to the next
   prefix.
3. After 5 prefixes, raise `RuntimeError` with a clear message.

Keep the loading logic in a helper that returns `None` for "unusable,
redraw", so it can be tested on its own. Close files in a `finally` block
(or use `with`).

## Field reading and preprocessing

- **NaNs**: Yoke simulation fields are NaN outside a material. Replace them
  with `np.nan_to_num(a, nan=0.0)` at read time (see `_nan0` in
  `pagosa_2frame.py`), unless the user wants a mask channel.
- **Volume fractions**: density fields may need weighting by
  `vofm_<part>` (see `yoke.datasets.lsc_dataset.volfrac_density` and
  `yoke.datasets.pagosa_2frame.merge_material_parts`).
- **Geometry**: axisymmetric data is stored as a half image. A
  `half_image: bool` argument chooses between the half and a reflected full
  image (`np.concatenate((np.fliplr(img), img), axis=1)`, which doubles W).
- **Channels**: stack fields as `(C, H, W)` float32 in a fixed, documented
  order. If the set of channels differs between samples, also return a
  `channel_map` of indices into a fixed vocabulary (see
  `yoke.datasets.pagosa_fields.CANONICAL_FIELDS`). This forces
  `batch_size=1` unless batches are grouped by map.
- **Time**: prefer a stored time field (`sim_time`) for lead times over
  `index * dt`. Reject non-positive time steps.
- **Normalization**: either none (with a per-channel loss such as `LpLoss`),
  or a precomputed stats file applied at read time. Yoke's format is in
  `src/yoke/datasets/lsc_normalization.py`: an `.npz` with `channels` and
  per-channel `__family`/`__scale` entries, written by `save_channel_norm`,
  read by `load_channel_norm`, and applied by `normalize_field`. These
  transforms only scale, never shift. Store stats files in
  `applications/normalization/`, and compute them with a script there, not in
  the dataset.
- **Resizing/padding**: `yoke.datasets.transforms.ResizePadCrop` is used
  when a model needs a size the data doesn't have (for example
  `LodeRunner` patch/window divisibility).

## Return values

- Return a **tuple of tensors** matching a contract in
  `skills/shared/batch_contracts.md`. Use float32 for data, float32 for
  times, int64 for indices and ids.
- No batch dimension. The `DataLoader` adds it.
- Document the tuple in the class docstring (names, shapes, dtypes, order),
  as the module docstring of `pagosa_2frame.py` does.

## Mixing datasets

`yoke.datasets.mixed_dataset.MixedDataSet(datasets, weights, length=int(1e6))`
splits the index range into blocks in proportion to `weights` and sends each
index to one dataset. All members must return the same contract. Pair it
with the varchan contract when channel sets differ.

## Docstring and annotation checklist

- Module docstring: what data, which models or contract it feeds, and the
  return tuple.
- Class docstring with `Args:` for every constructor argument.
- `__init__`, `__len__`, `__getitem__`, and helpers fully annotated.
- `Raises:` sections for `ValueError`/`RuntimeError`/`KeyError` raised on
  bad input.
