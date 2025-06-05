"""Unit tests for the *lsc_dataset* classes.

We use the *mock* submodule of *unittest* to allow fake files, directories, and
data for testing. This avoids a lot of costly sample file storage.

"""

import os
import pytest
import tempfile
import numpy as np
import torch
from unittest.mock import patch, mock_open, MagicMock
from yoke.datasets.lsc_dataset import LSC_rho2rho_temporal_DataSet
from yoke.datasets.lsc_dataset import LSC_cntr2hfield_DataSet
from yoke.datasets.lsc_dataset import LSC_hfield_reward_DataSet
from yoke.datasets.lsc_dataset import LSC_hfield_policy_DataSet


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
    return np.nan_to_num(np.ones((10, 10)), nan=0.0)  # Return a simple array for testing


# For LSC_rho2rho_temporal_DataSet
@pytest.fixture
def r2r_temporal_dataset() -> LSC_rho2rho_temporal_DataSet:
    """Setup an instance of the dataset.

    Mock arguments are used for testing.

    """
    LSC_NPZ_DIR = "/mock/path/"
    file_prefix_list = "mock_file_prefix_list.txt"
    max_timeIDX_offset = 3
    max_file_checks = 5

    mock_file_list = "mock_prefix_1\nmock_prefix_2\nmock_prefix_3\n"
    with patch("builtins.open", mock_open(read_data=mock_file_list)):
        with patch("random.shuffle") as mock_shuffle:
            ds = LSC_rho2rho_temporal_DataSet(
                LSC_NPZ_DIR, file_prefix_list, max_timeIDX_offset, max_file_checks
            )
            mock_shuffle.assert_called_once()

    return ds


def test_r2r_temporal_dataset_init(
    r2r_temporal_dataset: LSC_rho2rho_temporal_DataSet,
) -> None:
    """Test that the dataset is initialized correctly."""
    assert r2r_temporal_dataset.LSC_NPZ_DIR == "/mock/path/"
    assert r2r_temporal_dataset.max_timeIDX_offset == 3
    assert r2r_temporal_dataset.max_file_checks == 5
    assert r2r_temporal_dataset.Nsamples == 3

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

    assert any(field in exp_fields for field in r2r_temporal_dataset.hydro_fields), (
        f"None of the expected hydro fields found. Expected some of {exp_fields}, "
        f"but got {set(r2r_temporal_dataset.hydro_fields)}"
    )


def test_r2r_temporal_len(r2r_temporal_dataset: LSC_rho2rho_temporal_DataSet) -> None:
    """Test that the dataset length is correctly returned."""
    assert len(r2r_temporal_dataset) == int(1e6)


@patch("yoke.datasets.lsc_dataset.LSCread_npz_NaN", side_effect=mock_LSCread_npz_NaN)
@patch(
    "numpy.load", side_effect=lambda _: MockNpzFile({"dummy_field": np.ones((10, 10))})
)
@patch("pathlib.Path.is_file", return_value=True)
def test_r2r_temporal_getitem(
    mock_is_file: MagicMock,
    mock_npz_load: MagicMock,
    mock_LSCread_npz_NaN: MagicMock,
    r2r_temporal_dataset: LSC_rho2rho_temporal_DataSet,
) -> None:
    """Test the retrieval of items from the dataset."""
    idx = 0
    start_img, end_img, Dt = r2r_temporal_dataset[idx]

    assert isinstance(start_img, torch.Tensor)
    assert isinstance(end_img, torch.Tensor)
    assert isinstance(Dt, torch.Tensor)

    assert start_img.shape == (8, 10, 10)
    assert end_img.shape == (8, 10, 10)


def test_r2r_temporal_file_prefix_list_loading(
    r2r_temporal_dataset: LSC_rho2rho_temporal_DataSet,
) -> None:
    """Test that the file prefix list is loaded correctly."""
    expected_prefixes = ["mock_prefix_1", "mock_prefix_2", "mock_prefix_3"]
    assert sorted(r2r_temporal_dataset.file_prefix_list) == sorted(expected_prefixes)


@patch("pathlib.Path.is_file", return_value=False)
def test_r2r_temporal_getitem_max_file_checks(
    mock_is_file: MagicMock, r2r_temporal_dataset: LSC_rho2rho_temporal_DataSet
) -> None:
    """Test that max_file_checks is respected.

    Ensure FileNotFoundError is raised if files are not found.

    """
    err_msg = (
        r"\[Errno 2\] No such file or directory: "
        r"'/mock/path/mock_prefix_2_pvi_idx\d{5}\.npz'"
    )
    with pytest.raises(FileNotFoundError, match=err_msg):
        r2r_temporal_dataset[0]


@patch("numpy.load", side_effect=OSError("File could not be loaded"))
def test_r2r_temporal_getitem_load_error(
    mock_npz_load: MagicMock, r2r_temporal_dataset: LSC_rho2rho_temporal_DataSet
) -> None:
    """Test error thrown if load unsuccessful."""
    with pytest.raises(IOError, match="File could not be loaded"):
        r2r_temporal_dataset[0]


# Tests for cntr2field dataset
@pytest.fixture
def create_cntr2field_mock_files() -> None:
    """Create temporary files and directories for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        npz_dir = os.path.join(tmpdir, "npz_files/")
        os.makedirs(npz_dir, exist_ok=True)

        npz_file = os.path.join(npz_dir, "test_file.npz")
        np.savez(npz_file, dummy_data=np.array([1, 2, 3]))

        filelist_path = os.path.join(tmpdir, "filelist.txt")
        with open(filelist_path, "w") as f:
            f.write("test_file.npz\n")

        design_file = os.path.join(tmpdir, "design.csv")
        with open(design_file, "w") as f:
            f.write("sim_key,bspline_node1,bspline_node2\n")
            f.write("test_key,0.1,0.2\n")

        yield {
            "npz_dir": npz_dir,
            "filelist": filelist_path,
            "design_file": design_file,
        }


@patch("yoke.datasets.lsc_dataset.LSCread_npz_NaN")
@patch("yoke.datasets.lsc_dataset.LSCnpz2key")
@patch("yoke.datasets.lsc_dataset.LSCcsv2bspline_pts")
def test_cntr2field_dataset_length(
    mock_lsc_csv2bspline_pts: MagicMock,
    mock_lsc_npz2key: MagicMock,
    mock_lsc_read_npz: MagicMock,
    create_cntr2field_mock_files: dict[str, str],
) -> None:
    """Test that the dataset length matches the number of samples."""
    files = create_cntr2field_mock_files
    dataset = LSC_cntr2hfield_DataSet(
        LSC_NPZ_DIR=files["npz_dir"],
        filelist=files["filelist"],
        design_file=files["design_file"],
        half_image=True,
    )

    assert len(dataset) == 1


@patch("yoke.datasets.lsc_dataset.LSCread_npz_NaN")
@patch("yoke.datasets.lsc_dataset.LSCnpz2key")
@patch("yoke.datasets.lsc_dataset.LSCcsv2bspline_pts")
def test_cntr2field_dataset_getitem(
    mock_lsc_csv2bspline_pts: MagicMock,
    mock_lsc_npz2key: MagicMock,
    mock_lsc_read_npz_NaN: MagicMock,
    create_cntr2field_mock_files: dict[str, str],
) -> None:
    """Test that the __getitem__ method returns the correct data format."""
    files = create_cntr2field_mock_files

    # Mock return values
    mock_lsc_read_npz_NaN.return_value = np.array([0.0, 0.0, 1.0])
    mock_lsc_npz2key.return_value = "test_key"
    mock_lsc_csv2bspline_pts.return_value = np.array([0.1, 0.2])

    dataset = LSC_cntr2hfield_DataSet(
        LSC_NPZ_DIR=files["npz_dir"],
        filelist=files["filelist"],
        design_file=files["design_file"],
        half_image=True,
    )

    geom_params, hfield = dataset[0]

    # Check types
    assert isinstance(geom_params, torch.Tensor)
    assert isinstance(hfield, torch.Tensor)

    # Check values
    assert geom_params.shape == (2,)
    assert hfield.shape == (1, 3)  # Assuming single channel


def test_cntr2field_invalid_filelist(
    create_cntr2field_mock_files: dict[str, str],
) -> None:
    """Test behavior with an invalid file list."""
    files = create_cntr2field_mock_files
    invalid_filelist = os.path.join(tempfile.gettempdir(), "invalid_filelist.txt")

    with pytest.raises(FileNotFoundError):
        LSC_cntr2hfield_DataSet(
            LSC_NPZ_DIR=files["npz_dir"],
            filelist=invalid_filelist,
            design_file=files["design_file"],
        )


def test_cntr2field_empty_dataset(create_cntr2field_mock_files: dict[str, str]) -> None:
    """Test behavior when file list is empty."""
    files = create_cntr2field_mock_files

    # Create an empty filelist
    with open(files["filelist"], "w") as f:
        f.truncate(0)

    dataset = LSC_cntr2hfield_DataSet(
        LSC_NPZ_DIR=files["npz_dir"],
        filelist=files["filelist"],
        design_file=files["design_file"],
    )

    assert len(dataset) == 0


# For LSC_hfield_reward_DataSet
@pytest.fixture
@patch("numpy.load")
@patch("yoke.datasets.lsc_dataset.LSCread_npz_NaN", side_effect=mock_LSCread_npz_NaN)
def mock_reward_dataset(
    mock_LSCread_npz_NaN_func: MagicMock,
    mock_np_load: MagicMock,
) -> LSC_hfield_reward_DataSet:
    """Fixture to create a mock instance of LSC_hfield_reward_DataSet."""
    LSC_NPZ_DIR = "/mock/path/"
    filelist = "mock_filelist.txt"
    design_file = "mock_design.csv"
    half_image = True
    field_list = ("density_throw",)

    reward_fn = MagicMock(return_value=torch.tensor(1.0))

    # Mock numpy.load to return a mock dictionary
    mock_np_load.return_value = MockNpzFile({"density_throw": np.array([1.0, 2.0, 3.0])})

    mock_file_list = "mock_file_1\nmock_file_2\nmock_file_3\n"
    with patch("builtins.open", mock_open(read_data=mock_file_list)):
        with patch("random.shuffle") as mock_shuffle:
            ds = LSC_hfield_reward_DataSet(
                LSC_NPZ_DIR, filelist, design_file, half_image, field_list, reward_fn
            )
            mock_shuffle.assert_called_once()

    return ds


def test_reward_init(mock_reward_dataset: LSC_hfield_reward_DataSet) -> None:
    """Test initialization of the dataset."""
    assert mock_reward_dataset.LSC_NPZ_DIR == "/mock/path/"
    assert mock_reward_dataset.filelist == ["mock_file_1", "mock_file_2", "mock_file_3"]
    # Cartesian product of two files
    assert len(mock_reward_dataset.state_target_list) == 9
    assert mock_reward_dataset.hydro_fields == ("density_throw",)
    assert callable(mock_reward_dataset.reward)


def test_reward_len(mock_reward_dataset: LSC_hfield_reward_DataSet) -> None:
    """Test the __len__ method."""
    assert len(mock_reward_dataset) == int(1e6)


@patch("yoke.datasets.lsc_dataset.LSCread_npz_NaN", side_effect=mock_LSCread_npz_NaN)
@patch("yoke.datasets.lsc_dataset.LSCnpz2key", return_value="mock_key")
@patch(
    "yoke.datasets.lsc_dataset.LSCcsv2bspline_pts",
    return_value=np.array([0.5, 0.6, 0.7]),
)
@patch(
    "numpy.load", side_effect=lambda _: MockNpzFile({"density_throw": np.ones((10, 10))})
)
@patch("pathlib.Path.is_file", return_value=True)
def test_reward_getitem(
    mock_is_file: MagicMock,
    mock_npz_load: MagicMock,
    mock_LSCread_npz_NaN: MagicMock,
    mock_lsc_csv2bspline_pts: MagicMock,
    mock_lsc_npz2key: MagicMock,
    mock_reward_dataset: LSC_hfield_reward_DataSet,
) -> None:
    """Test the __getitem__ method."""
    result = mock_reward_dataset[0]
    state_geom_params, state_hfield, target_hfield, reward = result

    assert state_geom_params.shape == torch.Size([3])  # Mocked B-spline node shape
    assert state_hfield.shape == torch.Size([1, 10, 10])
    assert target_hfield.shape == torch.Size([1, 10, 10])
    assert reward == torch.tensor(1.0)


@patch("yoke.datasets.lsc_dataset.LSCread_npz_NaN", side_effect=mock_LSCread_npz_NaN)
@patch("yoke.datasets.lsc_dataset.LSCnpz2key", return_value="mock_key")
@patch(
    "yoke.datasets.lsc_dataset.LSCcsv2bspline_pts",
    return_value=np.array([0.5, 0.6, 0.7]),
)
@patch(
    "numpy.load", side_effect=lambda _: MockNpzFile({"density_throw": np.ones((10, 10))})
)
@patch("pathlib.Path.is_file", return_value=True)
def test_reward_function_invocation(
    mock_is_file: MagicMock,
    mock_npz_load: MagicMock,
    mock_LSCread_npz_NaN: MagicMock,
    mock_lsc_csv2bspline_pts: MagicMock,
    mock_lsc_npz2key: MagicMock,
    mock_reward_dataset: LSC_hfield_reward_DataSet,
) -> None:
    """Test the reward function invocation."""
    mock_reward_fn = mock_reward_dataset.reward
    mock_reward_dataset[0]
    assert mock_reward_fn.called


# For LSC_hfield_policy_DataSet
@pytest.fixture
@patch("numpy.load")
@patch("yoke.datasets.lsc_dataset.LSCread_npz_NaN", side_effect=mock_LSCread_npz_NaN)
def mock_policy_dataset(
    mock_LSCread_npz_NaN_func: MagicMock,
    mock_np_load: MagicMock,
) -> LSC_hfield_policy_DataSet:
    """Fixture to create a mock instance of LSC_hfield_policy_DataSet."""
    LSC_NPZ_DIR = "/mock/path/"
    filelist = "mock_filelist.txt"
    design_file = "mock_design.csv"
    half_image = True
    field_list = ("density_throw",)

    # Mock numpy.load to return a mock dictionary
    mock_np_load.return_value = MockNpzFile({"density_throw": np.array([1.0, 2.0, 3.0])})

    mock_file_list = "mock_file_1\nmock_file_2\nmock_file_3\n"
    with patch("builtins.open", mock_open(read_data=mock_file_list)):
        with patch("random.shuffle") as mock_shuffle:
            ds = LSC_hfield_policy_DataSet(
                LSC_NPZ_DIR, filelist, design_file, half_image, field_list
            )
            mock_shuffle.assert_called_once()

    return ds


def test_policy_init(mock_policy_dataset: LSC_hfield_policy_DataSet) -> None:
    """Test initialization of the dataset."""
    assert mock_policy_dataset.LSC_NPZ_DIR == "/mock/path/"
    assert mock_policy_dataset.filelist == ["mock_file_1", "mock_file_2", "mock_file_3"]
    # Cartesian product of two files
    assert len(mock_policy_dataset.state_target_list) == 9
    assert mock_policy_dataset.hydro_fields == ("density_throw",)


def test_policy_len(mock_policy_dataset: LSC_hfield_policy_DataSet) -> None:
    """Test the __len__ method."""
    assert len(mock_policy_dataset) == int(1e6)


@patch("yoke.datasets.lsc_dataset.LSCread_npz_NaN", side_effect=mock_LSCread_npz_NaN)
@patch("yoke.datasets.lsc_dataset.LSCnpz2key", return_value="mock_key")
@patch(
    "yoke.datasets.lsc_dataset.LSCcsv2bspline_pts",
    return_value=np.array([0.5, 0.6, 0.7]),
)
@patch(
    "numpy.load", side_effect=lambda _: MockNpzFile({"density_throw": np.ones((10, 10))})
)
@patch("pathlib.Path.is_file", return_value=True)
def test_policy_getitem(
    mock_is_file: MagicMock,
    mock_npz_load: MagicMock,
    mock_LSCread_npz_NaN: MagicMock,
    mock_lsc_csv2bspline_pts: MagicMock,
    mock_lsc_npz2key: MagicMock,
    mock_policy_dataset: LSC_hfield_policy_DataSet,
) -> None:
    """Test the __getitem__ method."""
    result = mock_policy_dataset[0]
    state_geom_params, state_hfield, target_hfield, geom_discrepancy = result

    assert state_geom_params.shape == torch.Size([3])  # Mocked B-spline node shape
    assert state_hfield.shape == torch.Size([1, 10, 10])
    assert target_hfield.shape == torch.Size([1, 10, 10])
    assert torch.allclose(geom_discrepancy, torch.tensor([0.0, 0.0, 0.0]), atol=1e-6), (
        "Tensors are not equal."
    )
