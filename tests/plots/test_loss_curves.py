"""Tests for yoke.plots.loss_curves."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from yoke.plots import loss_curves


def _write_record(path: Path, epoch: int, n_batches: int, loss: float) -> None:
    """Write a fake Yoke record CSV with constant loss.

    Args:
        path (Path): Destination CSV path.
        epoch (int): Epoch value written to every row.
        n_batches (int): Number of batch rows to write.
        loss (float): Constant loss value written to every row.
    """
    lines = [f"{epoch}, {b}, {loss}" for b in range(n_batches)]
    path.write_text("\n".join(lines) + "\n")


def _make_study(base: Path, studyIDX: int, n_trn: int, n_val: int) -> None:
    """Create a study dir with one training and one validation record CSV.

    Args:
        base (Path): Base directory to create the study under.
        studyIDX (int): Study index.
        n_trn (int): Number of training batch rows.
        n_val (int): Number of validation batch rows.
    """
    study_dir = base / f"study_{studyIDX:03d}"
    study_dir.mkdir(parents=True)
    _write_record(
        study_dir / f"training_study{studyIDX:03d}_epoch0010.csv",
        epoch=10,
        n_batches=n_trn,
        loss=0.5,
    )
    _write_record(
        study_dir / f"validation_study{studyIDX:03d}_epoch0010.csv",
        epoch=10,
        n_batches=n_val,
        loss=0.4,
    )


def test_find_record_csvs_parses_epochs(tmp_path: Path) -> None:
    """find_record_csvs returns sorted files and parsed epochs."""
    study_dir = tmp_path / "study_001"
    study_dir.mkdir()
    for ep in (1, 20, 3):
        _write_record(
            study_dir / f"training_study001_epoch{ep:04d}.csv",
            epoch=ep,
            n_batches=4,
            loss=0.1,
        )

    files, epochs = loss_curves.find_record_csvs(str(tmp_path), 1, "training")

    assert len(files) == 3
    assert epochs == [1, 3, 20]


def test_find_record_csvs_inprogress_drops_last(tmp_path: Path) -> None:
    """inprogress=True drops the most recent training CSV."""
    study_dir = tmp_path / "study_002"
    study_dir.mkdir()
    for ep in (1, 2):
        _write_record(
            study_dir / f"training_study002_epoch{ep:04d}.csv",
            epoch=ep,
            n_batches=4,
            loss=0.1,
        )

    files, epochs = loss_curves.find_record_csvs(
        str(tmp_path), 2, "training", inprogress=True
    )

    assert epochs == [1]
    assert len(files) == 1


def test_find_record_csvs_inprogress_empty_ok(tmp_path: Path) -> None:
    """The inprogress flag on an empty study does not raise."""
    (tmp_path / "study_003").mkdir()

    files, epochs = loss_curves.find_record_csvs(
        str(tmp_path), 3, "training", inprogress=True
    )

    assert files == []
    assert epochs == []


def test_find_record_csvs_bad_kind(tmp_path: Path) -> None:
    """An invalid kind raises ValueError."""
    with pytest.raises(ValueError, match="kind must be"):
        loss_curves.find_record_csvs(str(tmp_path), 1, "bogus")


def test_load_record_csv(tmp_path: Path) -> None:
    """load_record_csv reads Epoch/Batch/Loss columns."""
    csv_path = tmp_path / "rec.csv"
    _write_record(csv_path, epoch=5, n_batches=3, loss=0.25)

    df = loss_curves.load_record_csv(str(csv_path))

    assert list(df.columns) == ["Epoch", "Batch", "Loss"]
    assert len(df) == 3
    assert df["Loss"].iloc[0] == 0.25


def test_compute_quantile_bands_shapes() -> None:
    """compute_quantile_bands returns positions and a (3, n) band array."""
    losses = np.arange(20, dtype=float)

    positions, bands = loss_curves.compute_quantile_bands(losses, 5)

    assert positions.shape == (4,)
    assert bands.shape == (3, 4)


def test_compute_quantile_bands_remainder_warns() -> None:
    """A non-divisible length warns and trims the remainder."""
    losses = np.arange(22, dtype=float)

    with pytest.warns(UserWarning, match="not divisible"):
        positions, bands = loss_curves.compute_quantile_bands(losses, 5)

    assert positions.shape == (4,)
    assert bands.shape == (3, 4)


def test_plot_loss_curves_returns_figure(tmp_path: Path) -> None:
    """plot_loss_curves returns a Figure with expected labels and ylim."""
    _make_study(tmp_path, 1, n_trn=40, n_val=20)

    fig = loss_curves.plot_loss_curves(
        str(tmp_path),
        1,
        nsamps_per_trn_pt=10,
        nsamps_per_val_pt=5,
        ylim=2.0,
    )

    ax = fig.gca()
    assert ax.get_ylabel() == "Loss"
    assert ax.get_xlabel() == "Evaluation Index"
    assert ax.get_ylim() == (0.0, 2.0)
    assert len(ax.lines) > 0


def test_save_or_show_figure_saves(tmp_path: Path) -> None:
    """save_or_show_figure writes the expected PNG and returns its path."""
    _make_study(tmp_path, 7, n_trn=40, n_val=20)
    fig = loss_curves.plot_loss_curves(
        str(tmp_path),
        7,
        nsamps_per_trn_pt=10,
        nsamps_per_val_pt=5,
    )
    savedir = tmp_path / "images"

    result = loss_curves.save_or_show_figure(fig, 7, savefig=True, savedir=str(savedir))

    expected = savedir / "study007_TandV_curve.png"
    assert result == str(expected)
    assert expected.exists()


def _write_tagged_study(base: Path, studyIDX: int) -> None:
    """Create a study whose records carry an lsc/cyl dataset column.

    Args:
        base (Path): Base directory to create the study under.
        studyIDX (int): Study index.
    """
    study_dir = base / f"study_{studyIDX:03d}"
    study_dir.mkdir(parents=True)
    for kind, n in (("training", 40), ("validation", 20)):
        rows = [
            f"10, {b}, {0.5 if b % 2 else 0.1}, {'cyl' if b % 2 else 'lsc'}"
            for b in range(n)
        ]
        path = study_dir / f"{kind}_study{studyIDX:03d}_epoch0010.csv"
        path.write_text("\n".join(rows) + "\n")


def test_load_record_csv_tagged(tmp_path: Path) -> None:
    """Four-column records gain a Dataset column."""
    csv_path = tmp_path / "rec.csv"
    csv_path.write_text("1, 0, 0.5, lsc\n1, 0, 0.25, cyl\n")

    df = loss_curves.load_record_csv(str(csv_path))

    assert list(df.columns) == ["Epoch", "Batch", "Loss", "Dataset"]
    assert list(df["Dataset"]) == ["lsc", "cyl"]


def test_load_record_csv_bad_columns(tmp_path: Path) -> None:
    """Files with neither 3 nor 4 columns raise."""
    csv_path = tmp_path / "rec.csv"
    csv_path.write_text("1, 0\n")
    with pytest.raises(ValueError, match="expected 3 or 4 columns"):
        loss_curves.load_record_csv(str(csv_path))


def test_plot_loss_curves_dataset_filter(tmp_path: Path) -> None:
    """dataset= keeps only matching records; untagged files then raise."""
    _write_tagged_study(tmp_path, 2)

    fig = loss_curves.plot_loss_curves(
        str(tmp_path), 2, nsamps_per_trn_pt=10, nsamps_per_val_pt=5, dataset="cyl"
    )
    medians = fig.gca().lines[1].get_ydata()
    assert np.allclose(medians, 0.5)

    _make_study(tmp_path, 3, n_trn=40, n_val=20)
    with pytest.raises(ValueError, match="no Dataset column"):
        loss_curves.plot_loss_curves(
            str(tmp_path), 3, nsamps_per_trn_pt=10, nsamps_per_val_pt=5, dataset="lsc"
        )
