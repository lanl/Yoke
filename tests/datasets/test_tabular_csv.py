"""Tests for the row-per-sample CSV dataset and its k-fold splitter.

All data is a small synthetic CSV written to ``tmp_path``.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from yoke.datasets import tabular_csv as tc

INPUTS = ["a", "b"]
TARGETS = ["y"]


def _csv(tmp_path: Path, n: int = 15) -> Path:
    """Write an ``n``-row table with string, bool, and float columns.

    Rows 0 and 1 have ``ok=False``; ``grp`` alternates ``base``/``trans`` in a
    1:2 pattern; ``const`` is zero everywhere.
    """
    i = np.arange(n, dtype=float)
    df = pd.DataFrame(
        {
            "name": [f"r{k:02d}" for k in range(n)],
            "grp": ["base" if k % 3 == 0 else "trans" for k in range(n)],
            "ok": [k >= 2 for k in range(n)],
            "a": 1.0 + i,
            "b": 10.0 ** (i / 4.0),
            "const": np.zeros(n),
            "y": 2.0 * i + 3.0,
        }
    )
    path = tmp_path / "table.csv"
    df.to_csv(path, index=False)
    return path


def _ds(path: Path, **kwargs: object) -> tc.TabularCSV_DataSet:
    """Build a dataset with the default test columns."""
    return tc.TabularCSV_DataSet(str(path), INPUTS, TARGETS, **kwargs)


def test_kfold_partitions_rows() -> None:
    """Holdout folds are disjoint and together cover every row once."""
    seen = []
    for fold in range(5):
        train, hold = tc.kfold_indices(23, 5, fold, seed=3)
        assert np.intersect1d(train, hold).size == 0
        assert np.array_equal(np.union1d(train, hold), np.arange(23))
        seen.append(hold)
    allhold = np.concatenate(seen)
    assert np.array_equal(np.sort(allhold), np.arange(23))
    assert sorted(len(h) for h in seen) == [4, 4, 5, 5, 5]


def test_kfold_seed_determinism() -> None:
    """The same seed repeats a split and a different seed changes it."""
    _, h0 = tc.kfold_indices(40, 5, 0, seed=0)
    _, h0_again = tc.kfold_indices(40, 5, 0, seed=0)
    _, h1 = tc.kfold_indices(40, 5, 0, seed=1)
    assert np.array_equal(h0, h0_again)
    assert not np.array_equal(h0, h1)


def test_kfold_stratified_balance() -> None:
    """Fold sizes and per-stratum counts differ by at most one."""
    strata = np.array(["base"] * 70 + ["transition"] * 28)
    sizes, base_counts = [], []
    for fold in range(5):
        _, hold = tc.kfold_indices(98, 5, fold, seed=0, strata=strata)
        sizes.append(len(hold))
        base_counts.append(int(np.sum(strata[hold] == "base")))
    assert max(sizes) - min(sizes) <= 1
    assert max(base_counts) - min(base_counts) <= 1
    assert sum(base_counts) == 70


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_folds": 1, "fold": 0}, "n_folds"),
        ({"n_folds": 5, "fold": 5}, "fold must be"),
        ({"n_folds": 5, "fold": 0, "strata": np.zeros(3)}, "strata"),
    ],
)
def test_kfold_bad_args(kwargs: dict, match: str) -> None:
    """Invalid fold arguments raise ValueError."""
    with pytest.raises(ValueError, match=match):
        tc.kfold_indices(10, seed=0, **kwargs)


def test_kfold_too_few_rows() -> None:
    """Fewer rows than folds raises ValueError."""
    with pytest.raises(ValueError, match="Cannot split"):
        tc.kfold_indices(3, 5, 0, seed=0)


def test_contract_and_lengths(tmp_path: Path) -> None:
    """Items are float32 vectors; train and holdout complement each other."""
    path = _csv(tmp_path)
    train = _ds(path, split="train", n_folds=5, fold=2, split_seed=7)
    hold = _ds(path, split="holdout", n_folds=5, fold=2, split_seed=7)
    inpt, truth = train[0]
    assert inpt.shape == (2,) and truth.shape == (1,)
    assert inpt.dtype == torch.float32 and truth.dtype == torch.float32
    assert len(train) + len(hold) == 15
    assert not set(train.row_names) & set(hold.row_names)


def test_require_true_filters_rows(tmp_path: Path) -> None:
    """Rows failing a require_true column are dropped before splitting."""
    path = _csv(tmp_path)
    train = _ds(path, split="train", require_true=["ok"])
    hold = _ds(path, split="holdout", require_true=["ok"])
    names = set(train.row_names) | set(hold.row_names)
    assert len(names) == 13
    assert "r00" not in names and "r01" not in names


def test_stratify_col(tmp_path: Path) -> None:
    """Each holdout fold gets one or two of the five base rows."""
    path = _csv(tmp_path)
    for fold in range(3):
        hold = _ds(path, split="holdout", n_folds=3, fold=fold, stratify_col="grp")
        n_base = sum(int(n[1:]) % 3 == 0 for n in hold.row_names)
        assert n_base in (1, 2)


def test_train_split_is_standardized(tmp_path: Path) -> None:
    """Train-split columns have mean 0 and std 1 (ddof=0)."""
    train = _ds(_csv(tmp_path), split="train")
    assert torch.allclose(train.inputs.mean(0), torch.zeros(2), atol=1e-5)
    assert torch.allclose(train.inputs.std(0, unbiased=False), torch.ones(2))
    assert torch.allclose(train.targets.mean(0), torch.zeros(1), atol=1e-5)


def test_holdout_uses_train_stats(tmp_path: Path) -> None:
    """Holdout values are z-scored with the train fold's mean and std."""
    path = _csv(tmp_path)
    raw = pd.read_csv(path)
    train_idx, hold_idx = tc.kfold_indices(15, 5, 1, seed=0)
    y = raw["y"].to_numpy()
    expected = (y[hold_idx] - y[train_idx].mean()) / y[train_idx].std()
    hold = _ds(path, split="holdout", fold=1)
    assert np.allclose(hold.targets[:, 0].numpy(), expected, atol=1e-5)
    assert hold.target_mean.item() == pytest.approx(y[train_idx].mean())


def test_log_cols_and_no_normalize(tmp_path: Path) -> None:
    """log_cols applies log10; normalize=False returns the raw values."""
    path = _csv(tmp_path)
    ds = _ds(path, split="train", log_cols=["b"], normalize=False)
    names = [int(n[1:]) for n in ds.row_names]
    assert np.allclose(ds.inputs[:, 1].numpy(), np.array(names) / 4.0, atol=1e-6)
    assert np.allclose(ds.inputs[:, 0].numpy(), 1.0 + np.array(names))


def test_row_names_fallback(tmp_path: Path) -> None:
    """Without the id column, row_names are filtered-table row numbers."""
    ds = _ds(_csv(tmp_path), split="holdout", id_col="missing_id")
    assert all(n.isdigit() for n in ds.row_names)


def test_missing_column(tmp_path: Path) -> None:
    """A column not in the CSV raises KeyError naming it."""
    with pytest.raises(KeyError, match="nope"):
        tc.TabularCSV_DataSet(str(_csv(tmp_path)), ["a", "nope"], TARGETS)


@pytest.mark.parametrize(
    ("args", "kwargs", "match"),
    [
        ((["a", "const"], TARGETS), {}, "constant"),
        ((["a", "name"], TARGETS), {}, "numeric"),
        ((["a", "ok"], TARGETS), {}, "numeric"),
        ((INPUTS, TARGETS), {"split": "test"}, "split must be"),
        ((INPUTS, TARGETS), {"require_true": ["a"]}, "not boolean"),
        ((["a", "const"], TARGETS), {"log_cols": ["const"]}, "<= 0"),
        ((INPUTS, TARGETS), {"log_cols": ["const"]}, "log_cols must be"),
    ],
)
def test_bad_inputs(tmp_path: Path, args: tuple, kwargs: dict, match: str) -> None:
    """Invalid columns or options raise ValueError."""
    with pytest.raises(ValueError, match=match):
        tc.TabularCSV_DataSet(str(_csv(tmp_path)), *args, **kwargs)


def test_no_rows_left(tmp_path: Path) -> None:
    """Filtering away every row raises ValueError."""
    path = tmp_path / "t.csv"
    pd.DataFrame(
        {"a": [1.0, 2.0], "b": [1.0, 3.0], "y": [0.0, 1.0], "ok": [False, False]}
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="No rows"):
        _ds(path, require_true=["ok"])
