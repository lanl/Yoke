# Testing a New Dataset

Model new tests on `tests/datasets/test_pagosa_2frame.py`, which writes
small real files into `tmp_path`. Avoid the older style in
`tests/datasets/test_lsc_datasets.py`, which patches `open`/`np.load` with
mocks. That style is brittle: `mock_open` clashes with `np.load`.

## Rules

- Tests must pass with `pytest -Werror`. Any warning fails the suite.
- Plain test functions, each with a one-line docstring and `-> None`; annotate
  fixtures and arguments (`tmp_path: Path`).
- No real data and no network: build tiny synthetic files in the **same
  format** as the real data, with images a few pixels wide (for example
  `H, W = 4, 3`).
- There is no shared `conftest.py`, so define helpers in the test file.
- Import the module under test with a short alias:
  `from yoke.datasets import <module> as m`.

## What to test

1. **Contract**: the tuple has the right length, order, shapes, and dtypes.
2. **Values**: one known value per output (for example a constant field
   comes through, a NaN outside a material becomes 0, lead times equal the
   differences of the stored times).
3. **Determinism**: inject a fixed RNG and check which files were used.
4. **Missing files**: a missing frame causes a redraw or the next prefix. If
   nothing usable is left, it raises the documented error (`pytest.raises(...,
   match=...)`).
5. **Bad input**: constructor validation (empty filelist, bad offsets), and
   missing keys or columns.
6. **Options**: each flag changes the output as documented (for example
   `half_image=False` doubles W and mirrors).
7. **`__len__`**: `int(1e6)` for random sampling, N for fixed-index.

## Fixed RNG

```python
class _FixedRNG:
    """Stand-in RNG returning queued integers, in order."""

    def __init__(self, values: list[int]) -> None:
        """Store the queue of values to return."""
        self.values = list(values)

    def integers(self, low: int, high: int, endpoint: bool = False) -> int:
        """Return the next queued value, ignoring the bounds."""
        return self.values.pop(0)


ds._rng = _FixedRNG([1, 1, 1])  # e.g. gap_in, gap_out, start index
```

## Synthetic files per format

npz:

```python
def _frame(path: Path, sim_time: float, **fields: np.ndarray) -> None:
    """Write one synthetic frame with a scalar sim_time."""
    np.savez(path, sim_time=np.array(sim_time), **fields)
```

HDF5:

```python
import h5py
with h5py.File(tmp_path / "run.h5", "w") as f:
    f.create_dataset("fields/density", data=np.ones((5, H, W), np.float32))
    f.attrs["dt"] = 0.25
```

CSV/TXT/DAT:

```python
(tmp_path / "table.csv").write_text("id, a, b\n 1, 0.5, 2.0\n 2, 1.5, 3.0\n")
# include the quirks of the real file (leading spaces, comments, units row)
```

Images (in package code, read them with `skimage.io`; in tests, write them
with `skimage.io.imsave` or `tifffile.imwrite` if the dataset uses it):

```python
from skimage import io
io.imsave(tmp_path / "img_0000.tif", np.zeros((H, W), np.uint16),
          check_contrast=False)   # check_contrast=False avoids a warning
```

Filelists:

```python
def _prefix_file(tmp_path: Path, *prefixes: str) -> Path:
    """Write a prefix list and return its path."""
    p = tmp_path / "prefixes.txt"
    p.write_text("".join(f"{x}\n" for x in prefixes))
    return p
```

## Running

```bash
FILES="src/yoke/datasets/<module>.py tests/datasets/test_<module>.py"
<python> -m pytest -Werror tests/datasets/test_<module>.py -q
<ruff> check $FILES
<ruff> check --preview $FILES
<ruff> format --check --diff $FILES
```

CI lints `src/yoke/datasets` and `tests` as blocking steps, so both must be
clean.
