"""Unit tests for the *diffusion_dataset* classes.

We use the *mock* submodule of *unittest* to allow fake files, directories, and
data for testing. This avoids a lot of costly sample file storage.

"""

import pytest
import numpy as np
import torch
from unittest.mock import patch, mock_open, MagicMock
from yoke.datasets.diffusion_dataset import DiffusionLSC_temporal_DataSet
from yoke.utils.diffusion.noise_schedulers import VPCosineNoiseSchedule


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


# Mock LSCread_npz_NaN
def mock_LSCread_npz_NaN(npz_file: MockNpzFile, hfield: str) -> np.ndarray:
    """Test function to read data and replace NaNs with 0.0."""
    return np.nan_to_num(np.ones((10, 10)), nan=0.0)


# Mock volfrac_density
def mock_volfrac_density(
    img: np.ndarray, npz_file: MockNpzFile, hfield: str
) -> np.ndarray:
    """Test function to return image unchanged."""
    return img


@pytest.fixture
def diffusion_temporal_dataset() -> DiffusionLSC_temporal_DataSet:
    """Setup an instance of the diffusion dataset.

    Mock arguments are used for testing.

    """
    LSC_NPZ_DIR = "/mock/path/"
    file_prefix_list = "mock_file_prefix_list.txt"
    max_timeIDX_offset = 3
    max_file_checks = 5

    mock_file_list = "mock_prefix_1\nmock_prefix_2\nmock_prefix_3\n"
    with patch("builtins.open", mock_open(read_data=mock_file_list)):
        with patch("random.shuffle") as mock_shuffle:
            ds = DiffusionLSC_temporal_DataSet(
                LSC_NPZ_DIR, file_prefix_list, max_timeIDX_offset, max_file_checks
            )
            mock_shuffle.assert_called_once()

    return ds


def test_diffusion_temporal_dataset_init(
    diffusion_temporal_dataset: DiffusionLSC_temporal_DataSet,
) -> None:
    """Test that the dataset is initialized correctly."""
    assert diffusion_temporal_dataset.LSC_NPZ_DIR == "/mock/path/"
    assert diffusion_temporal_dataset.max_timeIDX_offset == 3
    assert diffusion_temporal_dataset.max_file_checks == 5
    assert diffusion_temporal_dataset.Nsamples == 3
    assert diffusion_temporal_dataset.half_image is True

    exp_fields = {
        "density_case",
        "density_cushion",
        "density_maincharge",
        "density_outside_air",
        "density_striker",
        "density_throw",
        "Uvelocity",
        "Wvelocity",
    }

    assert any(field in exp_fields for field in diffusion_temporal_dataset.in_vars), (
        f"None of the expected input fields found. Expected some of {exp_fields}, "
        f"but got {set(diffusion_temporal_dataset.in_vars)}"
    )

    # Test that out_vars defaults to in_vars
    assert np.array_equal(
        diffusion_temporal_dataset.in_vars, diffusion_temporal_dataset.out_vars
    )


def test_diffusion_temporal_dataset_custom_vars() -> None:
    """Test that custom in_vars and out_vars are set correctly."""
    LSC_NPZ_DIR = "/mock/path/"
    file_prefix_list = "mock_file_prefix_list.txt"
    max_timeIDX_offset = 3
    max_file_checks = 5

    in_vars = np.array(["density_case", "Uvelocity", "Wvelocity"])
    out_vars = np.array(["Uvelocity", "Wvelocity"])

    mock_file_list = "mock_prefix_1\n"
    with patch("builtins.open", mock_open(read_data=mock_file_list)):
        with patch("random.shuffle"):
            ds = DiffusionLSC_temporal_DataSet(
                LSC_NPZ_DIR,
                file_prefix_list,
                max_timeIDX_offset,
                max_file_checks,
                in_vars=in_vars,
                out_vars=out_vars,
            )

    assert np.array_equal(ds.in_vars, in_vars)
    assert np.array_equal(ds.out_vars, out_vars)
    assert len(ds.in_vars) == 3
    assert len(ds.out_vars) == 2


def test_diffusion_temporal_len(
    diffusion_temporal_dataset: DiffusionLSC_temporal_DataSet,
) -> None:
    """Test that the dataset length is correctly returned."""
    assert len(diffusion_temporal_dataset) == int(1e6)


@patch(
    "yoke.datasets.diffusion_dataset.volfrac_density", side_effect=mock_volfrac_density
)
@patch(
    "yoke.datasets.diffusion_dataset.LSCread_npz_NaN", side_effect=mock_LSCread_npz_NaN
)
@patch(
    "numpy.load", side_effect=lambda _: MockNpzFile({"dummy_field": np.ones((10, 10))})
)
@patch("pathlib.Path.is_file", return_value=True)
def test_diffusion_temporal_getitem(
    mock_is_file: MagicMock,
    mock_npz_load: MagicMock,
    mock_LSCread_npz_NaN: MagicMock,
    mock_volfrac_density: MagicMock,
    diffusion_temporal_dataset: DiffusionLSC_temporal_DataSet,
) -> None:
    """Test the retrieval of items from the dataset."""
    idx = 0
    x, y_tau, noise, lead_time, tau = diffusion_temporal_dataset[idx]

    # Check types
    assert isinstance(x, torch.Tensor)
    assert isinstance(y_tau, torch.Tensor)
    assert isinstance(noise, torch.Tensor)
    assert isinstance(lead_time, torch.Tensor)
    assert isinstance(tau, torch.Tensor)

    # Check shapes - 8 channels (default hydro fields)
    assert x.shape == (8, 10, 10)
    assert y_tau.shape == (8, 10, 10)
    assert noise.shape == (8, 10, 10)

    # Check scalar tensors
    assert lead_time.shape == ()
    assert tau.shape == ()

    # Check value ranges
    assert 0.0 <= tau.item() <= 1.0
    assert lead_time.item() >= 0.0


@patch(
    "yoke.datasets.diffusion_dataset.volfrac_density", side_effect=mock_volfrac_density
)
@patch(
    "yoke.datasets.diffusion_dataset.LSCread_npz_NaN", side_effect=mock_LSCread_npz_NaN
)
@patch(
    "numpy.load", side_effect=lambda _: MockNpzFile({"dummy_field": np.ones((10, 10))})
)
@patch("pathlib.Path.is_file", return_value=True)
def test_diffusion_temporal_different_in_out_vars(
    mock_is_file: MagicMock,
    mock_npz_load: MagicMock,
    mock_LSCread_npz_NaN: MagicMock,
    mock_volfrac_density: MagicMock,
) -> None:
    """Test dataset with different input and output variables."""
    LSC_NPZ_DIR = "/mock/path/"
    file_prefix_list = "mock_file_prefix_list.txt"
    max_timeIDX_offset = 3
    max_file_checks = 5

    in_vars = np.array(["density_case", "Uvelocity", "Wvelocity"])
    out_vars = np.array(["Uvelocity", "Wvelocity"])

    mock_file_list = "mock_prefix_1\n"
    with patch("builtins.open", mock_open(read_data=mock_file_list)):
        with patch("random.shuffle"):
            ds = DiffusionLSC_temporal_DataSet(
                LSC_NPZ_DIR,
                file_prefix_list,
                max_timeIDX_offset,
                max_file_checks,
                in_vars=in_vars,
                out_vars=out_vars,
            )

    x, y_tau, noise, lead_time, tau = ds[0]

    # Input should have 3 channels, output should have 2
    assert x.shape == (3, 10, 10)
    assert y_tau.shape == (2, 10, 10)
    assert noise.shape == (2, 10, 10)


def test_diffusion_temporal_file_prefix_list_loading(
    diffusion_temporal_dataset: DiffusionLSC_temporal_DataSet,
) -> None:
    """Test that the file prefix list is loaded correctly."""
    expected_prefixes = ["mock_prefix_1", "mock_prefix_2", "mock_prefix_3"]
    assert sorted(diffusion_temporal_dataset.file_prefix_list) == sorted(
        expected_prefixes
    )


@patch("pathlib.Path.is_file", return_value=False)
def test_diffusion_temporal_getitem_max_file_checks(
    mock_is_file: MagicMock, diffusion_temporal_dataset: DiffusionLSC_temporal_DataSet
) -> None:
    """Test that max_file_checks is respected.

    Ensure FileNotFoundError is raised if files are not found.

    """
    err_msg = (
        r"\[Errno 2\] No such file or directory: "
        r"'/mock/path/mock_prefix_\d+_pvi_idx\d{5}\.npz'"
    )
    with pytest.raises(FileNotFoundError, match=err_msg):
        diffusion_temporal_dataset[0]


@patch("numpy.load", side_effect=OSError("File could not be loaded"))
def test_diffusion_temporal_getitem_load_error(
    mock_npz_load: MagicMock, diffusion_temporal_dataset: DiffusionLSC_temporal_DataSet
) -> None:
    """Test error thrown if load unsuccessful."""
    with pytest.raises(IOError, match="File could not be loaded"):
        diffusion_temporal_dataset[0]


def test_diffusion_temporal_noise_schedule() -> None:
    """Test that custom noise schedule is used."""
    LSC_NPZ_DIR = "/mock/path/"
    file_prefix_list = "mock_file_prefix_list.txt"
    max_timeIDX_offset = 3
    max_file_checks = 5

    custom_schedule = VPCosineNoiseSchedule()

    mock_file_list = "mock_prefix_1\n"
    with patch("builtins.open", mock_open(read_data=mock_file_list)):
        with patch("random.shuffle"):
            ds = DiffusionLSC_temporal_DataSet(
                LSC_NPZ_DIR,
                file_prefix_list,
                max_timeIDX_offset,
                max_file_checks,
                noise_schedule=custom_schedule,
            )

    assert ds.noise_schedule is custom_schedule


def test_diffusion_temporal_half_image_false() -> None:
    """Test dataset with half_image=False."""
    LSC_NPZ_DIR = "/mock/path/"
    file_prefix_list = "mock_file_prefix_list.txt"
    max_timeIDX_offset = 3
    max_file_checks = 5

    mock_file_list = "mock_prefix_1\n"
    with patch("builtins.open", mock_open(read_data=mock_file_list)):
        with patch("random.shuffle"):
            ds = DiffusionLSC_temporal_DataSet(
                LSC_NPZ_DIR,
                file_prefix_list,
                max_timeIDX_offset,
                max_file_checks,
                half_image=False,
            )

    assert ds.half_image is False
