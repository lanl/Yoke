"""Tests for lsc240420 sequential dataset."""

# test_lsc_rho2rho_sequential_dataset.py
from pathlib import Path
from collections.abc import Generator

import numpy as np
import pytest
import torch
from pytest import MonkeyPatch

# Adjust these imports to match your actual module layout
from yoke.datasets.lsc_dataset import LSC_rho2rho_sequential_DataSet


# Mock np.load to simulate loading .npz files
class MockNpzFile:
    """Set up mock file load."""

    def __init__(self, data: dict[str, np.ndarray]) -> None:
        """Setup mock data."""
        self.data = data

    def __getitem__(self, item: str) -> np.ndarray:
        """Return single mock data sample."""
        return self.data[item]

    def close(self) -> None:
        """Close the file."""
        pass


# Fake np.load to return a dummy dict-like object
def mock_np_load(*args: tuple, **kwargs: dict) -> MockNpzFile:
    """Mock for NPZ file load."""
    data = {"some_data": np.array([0, 1, 2])}
    return MockNpzFile(data)


# Mock LSCread_npz_NaN to return a small 2D array
def mock_lscread_npz_nan(data_dict: dict, field: str) -> np.ndarray:
    """Mock reading a field and removing NaNs."""
    return np.ones((2, 2), dtype=np.float32)


class FakeRNG:
    """Mock for `np.rng`."""

    def integers(self, low: int, high: int) -> int:
        """Provide a fake RNG that always returns 0."""
        return 0


@pytest.fixture
def prefix_list_file(tmp_path: Path) -> Generator[Path, None, None]:
    """Creates a temporary prefix list file for testing."""
    file_path = tmp_path / "file_prefix_list.txt"
    prefixes = ["prefixA", "prefixB", "prefixC"]
    file_path.write_text("\n".join(prefixes))
    yield file_path


@pytest.fixture
def dataset(prefix_list_file: Path, tmp_path: Path) -> LSC_rho2rho_sequential_DataSet:
    """Creates an instance of the dataset with minimal valid args for testing."""
    # Create a dummy directory inside tmp_path so it actually exists
    real_lsc_dir = tmp_path / "dummy_npz_dir"
    real_lsc_dir.mkdir(parents=True, exist_ok=True)

    return LSC_rho2rho_sequential_DataSet(
        LSC_NPZ_DIR=str(real_lsc_dir),
        file_prefix_list=str(prefix_list_file),
        seq_len=3,
        half_image=True,
    )


def test_init(prefix_list_file: Path, tmp_path: Path) -> None:
    """Test that the dataset initializes correctly, reads prefixes, and shuffles them."""
    # Expect a FileNotFoundError if LSC_NPZ_DIR does not exist
    with pytest.raises(FileNotFoundError):
        _ = LSC_rho2rho_sequential_DataSet(
            LSC_NPZ_DIR="non_existent_path",
            file_prefix_list=str(prefix_list_file),
            seq_len=3,
            half_image=True,
        )

    # 2. Create a dummy directory inside tmp_path so it actually exists
    real_lsc_dir = tmp_path / "dummy_npz_dir"
    real_lsc_dir.mkdir(parents=True, exist_ok=True)

    # Use a dummy_npz_dir to focus on prefix reading
    dataset_obj = LSC_rho2rho_sequential_DataSet(
        LSC_NPZ_DIR=str(real_lsc_dir),
        file_prefix_list=str(prefix_list_file),
        seq_len=3,
        half_image=False,
    )
    # Confirm it loaded 3 prefixes
    assert dataset_obj.Nsamples == 3


def test_len(dataset: LSC_rho2rho_sequential_DataSet) -> None:
    """Test that __len__() reports the correct number of samples."""
    assert len(dataset) == 3


def test_getitem_valid_sequence(
    dataset: LSC_rho2rho_sequential_DataSet, monkeypatch: MonkeyPatch
) -> None:
    """Test successful retrieval of a valid sequence of frames.

    Overrides file existence, random index generation, and np.load.

    """
    # Always return True for file existence
    monkeypatch.setattr(Path, "is_file", lambda self: True)

    # Replace dataset.rng with your FakeRNG instance
    monkeypatch.setattr(dataset, "rng", FakeRNG())

    monkeypatch.setattr("numpy.load", mock_np_load)

    monkeypatch.setattr(
        "yoke.datasets.lsc_dataset.LSCread_npz_NaN", mock_lscread_npz_nan
    )

    img_seq, dt = dataset[0]
    # seq_len=3, hydro_fields=8 => shape = [3, 8, 2, 2]
    assert img_seq.shape == (3, 8, 2, 2)
    # Check dt is a scalar float tensor with value 0.25
    assert dt.shape == torch.Size([])
    assert float(dt) == 0.25


def test_getitem_index_wrap(
    dataset: LSC_rho2rho_sequential_DataSet, monkeypatch: MonkeyPatch
) -> None:
    """Test that the index wraps around correctly using modulo."""
    monkeypatch.setattr(Path, "is_file", lambda self: True)

    # Replace dataset.rng with your FakeRNG instance
    monkeypatch.setattr(dataset, "rng", FakeRNG())

    monkeypatch.setattr("numpy.load", mock_np_load)

    monkeypatch.setattr(
        "yoke.datasets.lsc_dataset.LSCread_npz_NaN", mock_lscread_npz_nan
    )

    # Dataset length=3 => index=5 effectively becomes index=2
    seq, _ = dataset[5]
    assert seq.shape == (3, 8, 2, 2)


def test_getitem_np_load_error(
    dataset: LSC_rho2rho_sequential_DataSet, monkeypatch: MonkeyPatch
) -> None:
    """Test that an exception in np.load triggers a RuntimeError in __getitem__."""
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    # Force an error on np.load

    def raise_io_error(*args: tuple, **kwargs: dict) -> None:
        raise OSError("Mock load error")

    monkeypatch.setattr("numpy.load", raise_io_error)

    with pytest.raises(RuntimeError) as excinfo:
        _ = dataset[0]
    assert "Error loading file" in str(excinfo.value)


def test_getitem_half_image_false(
    prefix_list_file: Path,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """Test that half_image=False results in horizontally reflected images."""
    # Create a dummy directory inside tmp_path so it actually exists
    real_lsc_dir = tmp_path / "dummy_npz_dir"
    real_lsc_dir.mkdir(parents=True, exist_ok=True)

    ds = LSC_rho2rho_sequential_DataSet(
        LSC_NPZ_DIR=str(real_lsc_dir),
        file_prefix_list=str(prefix_list_file),
        seq_len=2,
        half_image=False,
    )

    monkeypatch.setattr(Path, "is_file", lambda self: True)

    # Replace dataset.rng with your FakeRNG instance
    monkeypatch.setattr(ds, "rng", FakeRNG())

    monkeypatch.setattr("numpy.load", mock_np_load)

    monkeypatch.setattr(
        "yoke.datasets.lsc_dataset.LSCread_npz_NaN", mock_lscread_npz_nan
    )

    img_seq, _ = ds[0]
    # seq_len=2, hydro_fields=8 => shape = [2, 8, 2, 4]
    assert img_seq.shape == (2, 8, 2, 4)

    # Verify reflection on one field
    first_frame_first_field = img_seq[0, 0]

    # Original row [1, 2, 3] -> [3, 2, 1, 1, 2, 3]
    expected_row0 = torch.tensor([1, 1, 1, 1], dtype=torch.float32)
    assert torch.allclose(first_frame_first_field[0], expected_row0)
