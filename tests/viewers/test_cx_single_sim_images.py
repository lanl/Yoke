"""Unit tests for cx_single_sim_images utility functions.

- load_sim_data
- extract_densities
- generate_contour_plot.
"""

import tempfile
from pathlib import Path
import numpy as np
from applications.viewers.cx_single_sim_images import (
    load_sim_data,
    extract_densities,
    generate_contour_plot,
)

# Constants for test
TEST_VARIABLES = {
    "density_wall": 1.0,
    "density_U.DU": 2.0,
    "density_booster": 3.0,
    "density_maincharge": 4.0,
}
PLOT_EXTENT = (0, 400, 0, 1120)
VARIABLE_KEYS = list(TEST_VARIABLES.keys())


def create_test_npz(file_path: Path, shape: tuple[int, int] = (10, 10)) -> Path:
    """Create a test .npz file with known values and structure."""
    data = {
        key: np.full(shape, value, dtype=np.float64)
        for key, value in TEST_VARIABLES.items()
    }
    np.savez(file_path, **data)
    return file_path


def test_load_sim_data_reads_npz() -> None:
    """Test that load_sim_data correctly loads all expected arrays."""
    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = Path(tmpdir) / "test_sim.npz"
        create_test_npz(npz_path)

        loaded = load_sim_data(npz_path)
        assert isinstance(loaded, dict)
        assert set(loaded.keys()) == set(VARIABLE_KEYS)
        assert all(isinstance(v, np.ndarray) for v in loaded.values())
        assert loaded["density_U.DU"].shape == (10, 10)
        assert np.allclose(loaded["density_booster"], 3.0)


def test_extract_densities_returns_expected_means() -> None:
    """Test that extract_densities computes accurate means."""
    dummy_data = {
        key: np.full((5, 5), value, dtype=np.float64)
        for key, value in TEST_VARIABLES.items()
    }
    means = extract_densities(dummy_data, VARIABLE_KEYS)
    for key, expected in TEST_VARIABLES.items():
        assert key in means
        assert np.isclose(means[key], expected)


def test_generate_contour_plot_creates_file() -> None:
    """Test that generate_contour_plot creates a non-empty output image."""
    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = Path(tmpdir) / "test_sim.npz"
        output_path = Path(tmpdir) / "plot_output.png"
        create_test_npz(npz_path)

        data = load_sim_data(npz_path)
        generate_contour_plot(data, PLOT_EXTENT, output_path, VARIABLE_KEYS)

        assert output_path.exists()
        assert output_path.stat().st_size > 0
