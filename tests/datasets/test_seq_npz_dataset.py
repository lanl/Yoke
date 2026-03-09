"""Tests for the sequential NPZ dataset function."""

from pathlib import Path
import numpy as np
import torch
import pytest

import yoke.datasets.load_npz_dataset as dsmod
from yoke.datasets.load_npz_dataset import SequentialDataSet


class FakeLabeledData:
    """Stub for LabeledData with minimal required methods."""

    def __init__(self, npz: str, csv: str) -> None:
        """Initialize FakeLabeledData with npz and csv file paths."""
        pass

    def get_active_npz_field_names(self) -> list[str]:
        """Return a list of active NPZ field names."""
        return ["f1", "f2"]

    def get_active_hydro_field_names(self) -> list[str]:
        """Return a list of active hydro field names."""
        return ["H1", "H2"]

    def get_channel_map(self) -> list[int]:
        """Return the channel map as a list of integers."""
        return [0, 1]


@pytest.fixture(autouse=True)
def _stub_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub out LabeledData, import_img_from_npz, and process_channel_data."""
    # Replace LabeledData with our fake
    monkeypatch.setattr(dsmod, "LabeledData", FakeLabeledData)
    # import_img_from_npz returns a 1×1 array with value = len(fieldname)
    monkeypatch.setattr(
        dsmod,
        "import_img_from_npz",
        lambda npz, fld: np.full((1, 1), float(len(fld))),
    )
    # process_channel_data returns inputs unchanged
    monkeypatch.setattr(
        dsmod,
        "process_channel_data",
        lambda cmap, imgs, names: (cmap, imgs, names),
    )


def test_init_invalid_directory(tmp_path: Path) -> None:
    """__init__ should raise FileNotFoundError when npz_dir does not exist."""
    bad_dir = str(tmp_path / "nonexistent")
    prefix_file = tmp_path / "prefixes.txt"
    prefix_file.write_text("pfx\n")
    with pytest.raises(FileNotFoundError):
        _ = SequentialDataSet(
            bad_dir,
            "design.csv",
            str(prefix_file),
            max_file_checks=1,
            seq_len=1,
        )


def test_len_and_prefix_loading(tmp_path: Path) -> None:
    """__len__ returns number of prefixes, and they are loaded correctly."""
    npz_dir = tmp_path / "npzs"
    npz_dir.mkdir()
    prefix_file = tmp_path / "prefixes.txt"
    lines = ["A", "B", "C"]
    prefix_file.write_text("\n".join(lines) + "\n")
    ds = SequentialDataSet(
        str(npz_dir),
        "design.csv",
        str(prefix_file),
        max_file_checks=1,
        seq_len=1,
    )
    assert len(ds) == 3
    # set equality since shuffle may reorder
    assert set(ds.file_prefix_list) == set(lines)


def test_getitem_no_sequence_raises(tmp_path: Path) -> None:
    """__getitem__ should error if no valid sequence is found in max attempts."""
    npz_dir = tmp_path / "empty"
    npz_dir.mkdir()
    prefix_file = tmp_path / "pfx.txt"
    prefix_file.write_text("X\n")
    ds = SequentialDataSet(
        str(npz_dir),
        "design.csv",
        str(prefix_file),
        max_file_checks=1,
        seq_len=2,
    )
    with pytest.raises(RuntimeError) as exc:
        _ = ds[0]
    msg = str(exc.value)
    assert "Failed to find valid sequence for prefix" in msg


def test_getitem_success_sequence(tmp_path: Path) -> None:
    """__getitem__ returns a correct tensor sequence, dt, and channel_map."""
    # Prepare npz files for prefix 'sim'
    npz_dir = tmp_path / "data"
    npz_dir.mkdir()
    prefix = "sim"
    # Create sim_pvi_idx00000.npz ... idx00002.npz
    for i in range(3):
        f = npz_dir / f"{prefix}_pvi_idx{i:05d}.npz"
        np.savez(f, dummy=np.zeros((1, 1)))
    # Prefix list
    pf = tmp_path / "pfx.txt"
    pf.write_text(prefix + "\n")
    ds = SequentialDataSet(
        str(npz_dir),
        "design.csv",
        str(pf),
        max_file_checks=3,
        seq_len=3,
        half_image=True,
    )
    # Fix RNG so start_idx=0

    class FakeRNG:
        def integers(self, low: int, high: int) -> int:
            """Return a fixed start index of 0."""
            return 0

    ds.rng = FakeRNG()

    img_seq, dt, ch_map = ds[0]
    # Shapes: seq_len=3, fields=2, H=1, W=1
    assert isinstance(img_seq, torch.Tensor)
    assert img_seq.shape == (3, 2, 1, 1)
    # dt == 0.25
    assert isinstance(dt, torch.Tensor)
    assert float(dt) == pytest.approx(0.25)
    # channel_map from FakeLabeledData
    assert ch_map == [0, 1]


def test_half_image_false_reflect(tmp_path: Path) -> None:
    """When half_image=False, each frame's images are reflected horizontally."""
    npz_dir = tmp_path / "data2"
    npz_dir.mkdir()
    prefix = "pre"
    for i in range(2):
        f = npz_dir / f"{prefix}_pvi_idx{i:05d}.npz"
        # content doesn't matter; import is stubbed
        np.savez(f, dummy=np.zeros((1, 1)))
    pf = tmp_path / "pfx2.txt"
    pf.write_text(prefix + "\n")
    ds = SequentialDataSet(
        str(npz_dir),
        "design.csv",
        str(pf),
        max_file_checks=2,
        seq_len=2,
        half_image=False,
    )

    class FakeRNG:
        def integers(self, low: int, high: int) -> int:
            """Return a fixed start index of 0."""
            return 0

    ds.rng = FakeRNG()

    img_seq, dt, ch_map = ds[0]
    # fake_import returns [[len('f1')]]==2.0 and [[len('f2')]]==2.0
    # reflect makes each (1×1) → (1×2), so final shape is (2, 2, 1, 2)
    assert img_seq.shape == (2, 2, 1, 2)
    assert ch_map == [0, 1]
    # dt still 0.25
    assert float(dt) == pytest.approx(0.25)
