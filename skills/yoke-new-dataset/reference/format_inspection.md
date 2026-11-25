# Cheap Inspection Recipes by File Format

Use these to learn what a data file holds **without loading all of it**.
Always start with the file size (`ls -lh`, or `du -sh` for a directory) and
the number of files matching the pattern. Run the snippets with the Python
interpreter the user named, against one or two sample files. Report what you
find, then confirm your reading with the user.

What to report for every format:

- names of the arrays, fields, or columns, with shape and dtype;
- whether there are NaNs, infinities, or sentinel values (for example
  `-999`), checked on a small slice;
- value ranges on a slice (min/max) and units, if the file says;
- any scalar metadata (time, step index, simulation id, grid spacing);
- how the filename encodes sample, simulation, and time index.

## Readers and dependencies

`pyproject.toml` declares `numpy` (through `torch`), `h5py`, `pandas`, and
`scikit-image`. Code in `src/yoke` may use these.

`tifffile`, `imageio`, and `PIL` are usually importable too, but only because
`scikit-image` depends on them. They are **not** declared. Calling them from
`src/yoke` means adding a declared dependency, which needs the user's
approval. Prefer `skimage.io.imread` in package code. During *inspection*,
using an installed package is fine.

Anything else (`netCDF4`, `xarray`, `vtk`, `zarr`, ...) is not installed in a
standard Yoke environment. Ask the user before using it, even to inspect.

## npz / npy

```python
import numpy as np
with np.load(path, allow_pickle=False) as z:        # NpzFile is lazy per key
    for k in z.files:
        a = z[k]                                     # loads this one array only
        print(k, a.shape, a.dtype)
```

- For a single big `.npy`, use `np.load(path, mmap_mode="r")` and look at a
  slice: `a[:2]`, `np.isnan(a[:, :10]).any()`.
- 0-d arrays (`shape == ()`) hold scalars such as `sim_time`. Read them with
  `float(z["sim_time"])`.
- Never use `allow_pickle=True` on untrusted files.
- Seen in Yoke data: material fields are NaN outside the material, paired
  `vofm_<part>` volume fractions, and coordinates `Rcoord`/`Zcoord`.

## HDF5 (.h5, .hdf5)

```python
import h5py
def show(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(name, obj.shape, obj.dtype, obj.chunks, dict(obj.attrs))
    else:
        print(name + "/", dict(obj.attrs))
with h5py.File(path, "r") as f:
    print(dict(f.attrs))
    f.visititems(show)
    small = f["some/dataset"][:2]                    # read only a slice
```

- Note the layout: one file per sample, per time step, or a whole run in one
  file with a time axis. This decides how `__getitem__` indexes into it.
- Note chunking and compression. Reading across chunks is slow, which matters
  for random-access training.
- `h5py` file handles do not survive `fork` well. Datasets should open the
  file lazily in `__getitem__` (or once per worker), not in `__init__`.

## CSV / TXT / DAT (delimited or fixed-width text)

Look at the raw text first, because headers, delimiters, and comments vary:

```python
with open(path) as f:
    for _ in range(15):
        print(repr(f.readline()))
```

Then read a few rows:

```python
import pandas as pd
df = pd.read_csv(path, nrows=20, sep=None, engine="python")   # sniffs sep
print(df.dtypes); print(df.head())
```

- Check for a header line, comment character (`#`), delimiter (comma, tab,
  whitespace runs: `sep=r"\s+"`), fixed-width columns (`pd.read_fwf`), and
  units in the header.
- Check for **leading or trailing spaces** in headers and values (the real
  `design_cx241203_MASTER.csv` has them). Strip both in the dataset.
- For large text files, count lines with `wc -l` and do not read them whole.
- `.dat` can also be **raw binary**. If `repr()` shows unreadable bytes, see
  "Raw binary" below.
- A design or metadata table indexed by simulation id is usually read once in
  `__init__` with `pd.read_csv(..., index_col=0)`.

## TIFF / PNG / JPEG and other images

Header-only first:

```python
import tifffile                                      # installed, undeclared
with tifffile.TiffFile(path) as t:
    print(len(t.pages), t.pages[0].shape, t.pages[0].dtype)
    print(t.pages[0].tags.get("ImageDescription"))
```

```python
from PIL import Image                                # installed, undeclared
with Image.open(path) as im:
    print(im.size, im.mode, getattr(im, "n_frames", 1))
```

- Report bit depth (8/16/32-bit, float), channels (grayscale/RGB/
  multi-band), page or frame count (stacks or time series), and physical
  pixel size if tagged.
- In package code, `skimage.io.imread(path)` loads an image as a NumPy
  array using a declared dependency.
- Integer images usually need scaling (for example `/ 65535.0`). Ask what
  the values mean physically.

## Raw binary

Ask the user for the layout: header size, dtype, byte order, and array shape.
Then confirm it on one file:

```python
a = np.memmap(path, dtype="<f4", mode="r", offset=HEADER_BYTES, shape=SHAPE)
print(a[:2, :5])
```

Check that `file size == HEADER_BYTES + prod(SHAPE) * itemsize`.

## Unknown formats (netCDF, VTK, Exodus, zarr, ...)

1. Ask the user how the files are read today (a script, a tool, a package).
2. Check whether that package is importable in the user's environment. Do
   not install it.
3. If it is needed in `src/yoke`, propose adding it as a dependency and wait
   for approval. Otherwise, suggest converting to npz/HDF5 offline as a
   separate preprocessing step outside the dataset class.

## Questions to ask after inspecting

- Is every sample file the same shape, or do grids, channels, or lengths
  vary between simulations?
- Which fields are inputs, which are targets, and which are metadata?
- Are there missing or corrupt files, or gaps in time indices? How should
  they be handled: skip, redraw, or fail?
- Is there an existing filelist, design table, or normalization file?
