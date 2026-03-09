"""Tests for the Nested-Cylinder dataset."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import pytest

import yoke.datasets.nestedcyl_dataset as nc_loader


def test_npz2key_extracts_study_key() -> None:
    """Test that npz2key extracts the study key from .npz filepath.

    Args:
        None.

    Returns:
        None.
    """
    filepath = "some/dir/nc231213_S2_id0042_pvi_data.npz"
    key = nc_loader.npz2key(filepath)
    assert key == "nc231213_S2_id0042"


def test_csv2scalar_returns_correct_value(tmp_path: Path) -> None:
    """Test that csv2scalar extracts the correct scalar value.

    Args:
        tmp_path: pytest temporary directory fixture.

    Returns:
        None.
    """
    # Create design CSV with known values
    design = pd.DataFrame(
        {"ptw_scale": [0.5, 1.5]},
        index=["nc231213_S1_id0001", "nc231213_S1_id0002"],
    )
    csv_file = tmp_path / "design.csv"
    design.to_csv(csv_file)

    value = nc_loader.csv2scalar(str(csv_file), "nc231213_S1_id0002", "ptw_scale")
    assert isinstance(value, float)
    assert value == 1.5


def test_npz_pvi2field_slices_and_concats(tmp_path: Path) -> None:
    """Test that npz_pvi2field slices and concatenates the array.

    Args:
        tmp_path: pytest temporary directory fixture.

    Returns:
        None.
    """
    # Build dummy array of shape (1000, 300)
    data = np.arange(1000 * 300, dtype=float).reshape(1000, 300)
    npz_path = tmp_path / "test.npz"
    np.savez(npz_path, rho=data)

    npz = np.load(npz_path)
    pic = nc_loader.npz_pvi2field(npz, "rho")

    # After slicing rows [800:] and cols [:250], then concat, shape is (200, 500)
    assert pic.shape == (200, 500)

    # Ensure left half is mirror of right half
    right = pic[:, 250:]
    left = pic[:, :250]
    assert np.all(left == np.fliplr(right))


class TestPviSingleFieldDataSet:
    """Unit tests for PVI_SingleField_DataSet class."""

    @pytest.fixture(autouse=True)
    def setup_dataset(self, tmp_path: Path) -> None:
        """Create temp NPZ files, file list, and design CSV for testing.

        Args:
            tmp_path: pytest temporary directory fixture.

        Returns:
            None.
        """
        # Make a directory for NPZ files
        self.npz_dir = tmp_path / "npz"
        self.npz_dir.mkdir()
        # Define two study keys and generate .npz files
        self.keys = ["nc231213_S1_id0001", "nc231213_S1_id0002"]
        self.filenames = []
        for key in self.keys:
            fn = f"{key}_pvi.npz"
            arr = np.ones((1000, 300), dtype=float)
            np.savez(self.npz_dir / fn, rho=arr)
            self.filenames.append(fn)

        # Create filelist text file
        self.filelist_path = tmp_path / "files.txt"
        self.filelist_path.write_text("\n".join(self.filenames))

        # Create design CSV mapping keys to ptw_scale
        df = pd.DataFrame({"ptw_scale": [2.0, 3.0]}, index=self.keys)
        self.design_path = tmp_path / "design.csv"
        df.to_csv(self.design_path)

        # Initialize the dataset
        self.dataset = nc_loader.PVI_SingleField_DataSet(
            NC_NPZ_DIR=str(self.npz_dir) + os.sep,
            filelist=str(self.filelist_path),
            input_field="rho",
            predicted="ptw_scale",
            design_file=str(self.design_path),
        )

    def test_len_returns_number_of_samples(self) -> None:
        """Test that __len__ returns the correct sample count."""
        assert len(self.dataset) == len(self.keys)

    def test_getitem_returns_input_and_truth(self) -> None:
        """Test that __getitem__ returns tensor and correct truth."""
        img, truth = self.dataset[0]

        # Validate the input tensor
        assert isinstance(img, torch.Tensor)
        assert img.dtype == torch.float32
        assert img.shape == (1, 200, 500)

        # Validate the scalar truth value
        assert isinstance(truth, float)
        assert truth == 2.0
