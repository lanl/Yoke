"""Unit tests for the *lsc_dataset* classes.

We use the *mock* submodule of *unittest* to allow fake files, directories, and
data for testing. This avoids a lot of costly sample file storage.

"""

import pathlib
import pandas as pd
import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock
from yoke.datasets.load_npz_dataset import (
    process_channel_data,
    LabeledData,
    TemporalDataSet,
)


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
def mock_read_npz_NaN(npz_file: MockNpzFile, hfield: str) -> np.ndarray:
    """Test function to read data and replace NaNs with 0.0."""
    return np.nan_to_num(np.ones((10, 10)), nan=0.0)  # Return a simple array for testing


@pytest.fixture
def design_csv(tmp_path: pathlib.Path) -> str:
    """Create a mock CSV with Steel as wallMat and Water as backMat."""
    df = pd.DataFrame(
        {"wallMat": ["Steel"], "backMat": ["Water"]},
        index=["cx240420_id00001"],
    )
    path = tmp_path / "mock_design.csv"
    df.to_csv(path)
    return str(path)


def labeled_data(tmp_path: pathlib.Path) -> LabeledData:
    """Setup an instance of the dataset.

    Instantiate LabeledData with a mock design CSV (Air/Al) and
    a dummy NPZ path for study 'cx240420_id00001'.
    """
    # 1) write mock CSV
    csv_path = tmp_path / "design.csv"
    df = pd.DataFrame(
        {"wallMat": ["Air"], "backMat": ["Al"]},
        index=["cx240420_id00001"],
    )
    df.to_csv(csv_path)

    # 2) define dummy NPZ filepath (no actual file needed)
    npz_path = tmp_path / "cx240420_id00001_pvi_idx00000.npz"

    # 3) return the instantiated object
    return LabeledData(
        npz_filepath=str(npz_path),
        csv_filepath=str(csv_path),
    )


def test_init_sets_all_expected_attributes(tmp_path: pathlib.Path) -> None:
    """Test that the dataset is initialized correctly."""
    ld = labeled_data(tmp_path)

    # metadata attributes
    assert ld.npz_filepath.endswith("cx240420_id00001_pvi_idx00000.npz")
    assert ld.csv_filepath.endswith("design.csv")
    assert ld.kinematic_variables == "velocity"
    assert ld.thermodynamic_variables == "density"
    assert ld.study == "cx"
    assert ld.key == "cx240420_id00001"

    # full hydro‐field names
    all_names = ld.get_hydro_field_names()
    assert isinstance(all_names, list)
    assert all_names[:4] == ["Rcoord", "Zcoord", "Uvelocity", "Wvelocity"]
    assert len(all_names) == 46

    # active fields for default velocity + density
    expected_active = [
        "Uvelocity",
        "Wvelocity",
        "density_Air",
        "density_Al",
        "density_maincharge",
        "density_booster",
    ]
    got_active = list(ld.get_active_hydro_field_names())
    assert got_active == expected_active

    # channel_map indices should point into all_names
    expected_idx = [all_names.index(fld) for fld in expected_active]
    got_idx = ld.get_channel_map()
    assert got_idx == expected_idx


def test_get_active_hydro_indices_matches_channel_map(
    tmp_path: pathlib.Path,
) -> None:
    """Tests get_active_hydro_indices().

    Checks whether get_active_hydro_indices() returns the same list
    as get_channel_map(), since channel_map is defined by it.
    """
    ld = labeled_data(tmp_path)
    active_indices = ld.get_active_hydro_indices()
    channel_map = ld.get_channel_map()
    assert active_indices == channel_map


def test_cylex_data_loader_appends_pressure_fields(
    tmp_path: pathlib.Path,
) -> None:
    """Tess cylex_data_loader().

    Checks that cylex_data_loader() adds only pressure fields (no energy)
    when thermodynamic_variables='all'.
    """
    # Prepare CSV
    csv_path = tmp_path / "design.csv"
    df = pd.DataFrame(
        {"wallMat": ["Air"], "backMat": ["Al"]},
        index=["cx240420_id00001"],
    )
    df.to_csv(csv_path)
    npz_path = tmp_path / "cx240420_id00001_pvi_idx00000.npz"

    # Instantiate with all thermodynamic variables
    ld = LabeledData(
        npz_filepath=str(npz_path),
        csv_filepath=str(csv_path),
        kinematic_variables="velocity",
        thermodynamic_variables="all",
    )

    # Get the active NPZ and hydro field names
    active_npz = list(ld.get_active_npz_field_names())
    active_hydro = list(ld.get_active_hydro_field_names())

    # Pressure fields should be present
    assert "pressure_wall" in active_npz
    assert "pressure_Al" in active_npz
    assert "pressure_maincharge" in active_npz
    assert "pressure_booster" in active_npz

    assert "pressure_Air" in active_hydro
    assert "pressure_Al" in active_hydro
    assert "pressure_maincharge" in active_hydro
    assert "pressure_booster" in active_hydro

    # Energy fields should not be present
    assert not any(name.startswith("energy_") for name in active_npz)
    assert not any(name.startswith("energy_") for name in active_hydro)


def test_extract_letters_handles_various_strings() -> None:
    """Tests extract_letters().

    Checks that extract_letters returns the leading alphabetic segment
    before the first digit, or None if no match.
    """
    ld = object.__new__(LabeledData)
    assert ld.extract_letters("cx241203_id01250") == "cx"
    assert ld.extract_letters("ABC123DEF") == "ABC"
    assert ld.extract_letters("noDigitsHere") is None
    assert ld.extract_letters("123ABC") is None


def test_get_study_and_key_parses_filepath() -> None:
    """Tests get_study_and_key().

    Checks that get_study_and_key correctly sets .study and .key
    based on the NPZ filename.
    """
    ld = object.__new__(LabeledData)
    ld.get_study_and_key("path/to/cx241203_id01250_pvi_idx00000.npz")
    assert ld.study == "cx"
    assert ld.key == "cx241203_id01250"

    ld.get_study_and_key("/abs/AbcXYZ123_id00099_pvi_foo.npz")
    assert ld.study == "AbcXYZ"
    assert ld.key == "AbcXYZ123_id00099"


def test_get_hydro_field_names_returns_all(tmp_path: pathlib.Path) -> None:
    """Test that get_hydro_field_names returns the full hydro field list."""
    ld = labeled_data(tmp_path)
    all_names = ld.get_hydro_field_names()
    assert isinstance(all_names, list)
    assert all_names == ld.all_hydro_field_names
    assert len(all_names) == len(ld.all_hydro_field_names)


def test_get_channel_map_returns_channel_map(tmp_path: pathlib.Path) -> None:
    """Test that get_channel_map returns the same list as the channel_map attr."""
    ld = labeled_data(tmp_path)
    assert ld.get_channel_map() == ld.channel_map
    assert isinstance(ld.get_channel_map(), list)


def test_get_active_hydro_field_names(tmp_path: pathlib.Path) -> None:
    """Test that get_active_hydro_field_names returns the active hydro fields."""
    ld = labeled_data(tmp_path)
    expected_active = [
        "Uvelocity",
        "Wvelocity",
        "density_Air",
        "density_Al",
        "density_maincharge",
        "density_booster",
    ]
    assert list(ld.get_active_hydro_field_names()) == expected_active


def test_get_active_npz_field_names(tmp_path: pathlib.Path) -> None:
    """Test that get_active_npz_field_names returns the active NPZ field names."""
    ld = labeled_data(tmp_path)
    expected_npz = [
        "Uvelocity",
        "Wvelocity",
        "density_wall",
        "density_Al",
        "density_maincharge",
        "density_booster",
    ]
    assert list(ld.get_active_npz_field_names()) == expected_npz


def test_process_channel_data_combines_duplicates(
    tmp_path: pathlib.Path,
) -> None:
    """Tests process_channel_data().

    Checks that process_channel_data() merges duplicate channels correctly,
    yielding unique channel_map, combined images, and labels.
    """
    # Use LabeledData to obtain the true hydro field list
    ld = labeled_data(tmp_path)
    hydro_field_list = ld.get_hydro_field_names()

    # Create a channel_map with a duplicate (index 4 repeated)
    channel_map = [0, 4, 4]
    active_names = [hydro_field_list[i] for i in channel_map]

    # Build a dummy img_list_combined: 1 image, 3 channels, 1×1 pixels
    # Channel 0 → 9, first 4 → 0, second 4 → 5
    img_list_combined = np.array([[[[9]], [[0]], [[5]]]])

    out_map, out_imgs, out_names = process_channel_data(
        channel_map, img_list_combined, active_names
    )

    # Expect unique channels [0,4]
    np.testing.assert_array_equal(out_map, [0, 4])

    # Expect combined image shape (1,2,1,1)
    assert out_imgs.shape == (1, 2, 1, 1)

    # Channel 0 unchanged, channel 4 picks first non-zero (5)
    assert out_imgs[0, 0, 0, 0] == 9
    assert out_imgs[0, 1, 0, 0] == 5

    # Labels correspond to indices 0 and 4
    assert out_names == [hydro_field_list[0], hydro_field_list[4]]


@pytest.fixture
def temporal_dataset(tmp_path: pathlib.Path) -> TemporalDataSet:
    """Set up an instance of TemporalDataSet.

    Build a TemporalDataSet pointing at tmp_path/mock/path, with three real
    'cx' prefixes and a matching design CSV so LabeledData populates its fields.
    """
    # Create npz directory
    npz_root = tmp_path / "mock" / "path"
    npz_root.mkdir(parents=True, exist_ok=True)
    npz_dir = str(npz_root) + "/"

    # Write prefix list (cx study keys)
    prefixes = ["cx240420_id00001", "cx240420_id00002", "cx240420_id00003"]
    prefix_file = npz_root / "mock_file_prefix_list.txt"
    prefix_file.write_text("\n".join(prefixes) + "\n")

    # Write minimal design CSV with wallMat/backMat for each cx key
    csv_path = npz_root / "design.csv"
    df = pd.DataFrame(
        {"wallMat": ["Air", "Air", "Air"], "backMat": ["Al", "Al", "Al"]},
        index=prefixes,
    )
    df.to_csv(csv_path)

    return TemporalDataSet(
        npz_dir=npz_dir,
        csv_filepath=str(csv_path),
        file_prefix_list=str(prefix_file),
        max_time_idx_offset=3,
        max_file_checks=5,
        half_image=True,
    )


def test_init_temporal_dataset(temporal_dataset: TemporalDataSet) -> None:
    """__init__ should set attributes based on the mock arguments."""
    ds = temporal_dataset
    assert ds.npz_dir.endswith("mock/path/")
    assert ds.csv_filepath.endswith("design.csv")
    assert ds.max_time_idx_offset == 3
    assert ds.max_file_checks == 5
    assert ds.half_image is True
    assert len(ds.file_prefix_list) == 3
    assert ds.n_samples == 3


def test_temporal_dataset_len(temporal_dataset: TemporalDataSet) -> None:
    """__len__ should return the fixed size of 800000."""
    assert len(temporal_dataset) == 800000


def test_file_prefix_list_loading(temporal_dataset: TemporalDataSet) -> None:
    """Test that the file_prefix_list fixture loads the three cx prefixes correctly."""
    expected = ["cx240420_id00001", "cx240420_id00002", "cx240420_id00003"]
    assert sorted(temporal_dataset.file_prefix_list) == sorted(expected)


@patch("pathlib.Path.is_file", return_value=False)
def test_temporal_dataset_getitem_max_file_checks(
    mock_is_file: MagicMock, temporal_dataset: TemporalDataSet
) -> None:
    """Test that max_file_checks is respected and FileNotFoundError is raised."""
    with pytest.raises(FileNotFoundError):
        _ = temporal_dataset[0]  # type: ignore


@patch("pathlib.Path.is_file", return_value=True)
@patch("numpy.load", side_effect=OSError("load failed"))
def test_temporal_dataset_getitem_load_error(
    mock_npz: MagicMock,
    mock_is_file: MagicMock,
    temporal_dataset: TemporalDataSet,
) -> None:
    """Test that an OSError in numpy.load propagates out of __getitem__."""
    with pytest.raises(OSError, match="load failed"):
        _ = temporal_dataset[0]  # type: ignore


@patch("pathlib.Path.is_file", return_value=True)
@patch("numpy.load", side_effect=OSError("load failed"))
def test_getitem_propagates_load_error(
    mock_npz: MagicMock,
    mock_is_file: MagicMock,
    temporal_dataset: TemporalDataSet,
) -> None:
    """If numpy.load throws, __getitem__ should propagate that OSError."""
    with pytest.raises(OSError, match="load failed"):
        _ = temporal_dataset[0]  # type: ignore


@patch("pathlib.Path.is_file", return_value=True)
@patch("numpy.load", return_value=MagicMock())
@patch(
    "yoke.datasets.load_npz_dataset.import_img_from_npz",
    side_effect=lambda npz, fld: np.ones((10, 10)),
)
@patch(
    "yoke.datasets.load_npz_dataset.process_channel_data",
    side_effect=lambda cm, imgs, names: (cm, imgs, names),
)
def test_temporal_dataset_getitem_returns_expected(
    mock_proc: MagicMock,
    mock_import: MagicMock,
    mock_npz: MagicMock,
    mock_isfile: MagicMock,
    temporal_dataset: TemporalDataSet,
) -> None:
    """Tests that the __getitem__ method returns the correct data format as follows.

    - start_img, end_img: torch.Tensor of shape (n_ch,10,10)
    - cm1, cm2 matching LabeledData.get_channel_map()
    - dt = 0.25*(end_idx-start_idx)
    """
    ds = temporal_dataset

    # Make exactly two NPZ files so start_idx=0, end_idx=1 wrap cheaply
    prefix = ds.file_prefix_list[0]
    start = pathlib.Path(ds.npz_dir) / f"{prefix}_pvi_idx00000.npz"
    end = pathlib.Path(ds.npz_dir) / f"{prefix}_pvi_idx00001.npz"
    np.savez(start, dummy=np.zeros((1,)))
    np.savez(end, dummy=np.zeros((1,)))

    # Fix RNG so first integers→0, next→1
    class FakeRNG:
        cnt = 0

        def integers(self, low: int, high: int) -> int:
            self.cnt += 1
            return 0 if self.cnt == 1 else 1

    ds.rng = FakeRNG()

    # Fetch item
    start_img, cm1, end_img, cm2, dt = ds[0]  # type: ignore

    # Build expected channel map via LabeledData
    ld = LabeledData(str(start), ds.csv_filepath)
    expected_cm = ld.get_channel_map()

    # Channel maps
    assert cm1 == expected_cm
    assert cm2 == expected_cm

    # Tensors of shape (n_ch,10,10) filled with ones
    n_ch = len(expected_cm)
    assert isinstance(start_img, torch.Tensor)
    assert start_img.shape == (n_ch, 10, 10)
    assert torch.all(start_img == 1)

    assert isinstance(end_img, torch.Tensor)
    assert end_img.shape == (n_ch, 10, 10)
    assert torch.all(end_img == 1)

    # dt = 0.25*(1-0) == 0.25
    assert isinstance(dt, torch.Tensor)
    assert dt.item() == pytest.approx(0.25)
