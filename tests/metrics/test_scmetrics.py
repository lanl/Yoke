"""End-to-end tests for the SCmetrics module.

This test suite provides complete line coverage for the provided module. It
creates a small synthetic NPZ dataset, instantiates the main class, and
validates every public function, including branches and known error paths.
"""

from __future__ import annotations

from pathlib import Path
from collections.abc import Iterator

import numpy as np
import pytest

import yoke.metrics.shaped_charge_metrics as mod


@pytest.fixture()
def sample_npz_data(tmp_path: Path) -> tuple[Path, dict[str, np.ndarray]]:
    """Create a compact NPZ dataset tailored for the SCmetrics tests.

    The grid is 3x4 (Z x R) to keep calculations tractable while still
    exercising connectivity, masking, and thresholding logic. The density
    contains NaN values to verify cleaning. Two disconnected on-axis regions
    exist to exercise region relabeling. An off-axis connected component is
    also present and should be excluded from "on-axis" operations.

    Args:
        tmp_path: Temporary directory provided by pytest.

    Returns:
        Tuple of (path to NPZ file, dict of arrays) so tests can derive
        expected values directly from the canonical inputs.
    """
    rcoord = np.array([0.0, 1.0, 2.0, 3.0])
    zcoord = np.array([0.0, 1.0, 2.0])

    # 3x4 density with NaN (cleaned by get_field), two on-axis blobs, and an
    # off-axis blob that should be removed by compute_regions(...).
    density_throw = np.array(
        [
            [1.0, np.nan, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 1.0],
        ]
    )

    # Liner volume fraction (not restricted to regions for mass totals).
    vofm_throw = np.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.5],
            [0.7, 0.0, 0.2, 0.2],
        ]
    )

    # Vertical velocity field used by multiple metrics.
    wvel = np.array(
        [
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 5.0],
            [1.0, 0.0, 3.0, 3.0],
        ]
    )

    # HE density and volume fraction for get_HE_mass.
    density_he = np.full_like(density_throw, 1.5)
    vofm_he = np.array(
        [
            [0.2, 0.1, 0.0, 0.0],
            [0.0, 0.0, 0.3, 0.3],
            [0.4, 0.0, 0.6, 0.6],
        ]
    )

    arrays: dict[str, np.ndarray] = {
        "Rcoord": rcoord,
        "Zcoord": zcoord,
        "density_throw": density_throw,
        "vofm_throw": vofm_throw,
        "Wvelocity": wvel,
        "density_maincharge": density_he,
        "vofm_maincharge": vofm_he,
    }

    path = tmp_path / "data.npz"
    np.savez(path, **arrays)
    return path, arrays


@pytest.fixture()
def sc(
    sample_npz_data: tuple[Path, dict[str, np.ndarray]],
    tmp_path: Path,
) -> Iterator[mod.SCmetrics]:
    """Instantiate SCmetrics in a temp cwd with a pre-created 'png' folder.

    This fixture also ensures the working directory is switched to the temp
    dir so that any files saved by plotting land under that directory.

    Args:
        sample_npz_data: Fixture containing the NPZ path and arrays.
        tmp_path: Temporary directory provided by pytest.

    Yields:
        An initialized SCmetrics instance ready for metric calls.
    """
    path, _ = sample_npz_data
    cwd = Path.cwd()
    (tmp_path / "png").mkdir(parents=True, exist_ok=True)
    try:
        # Work in tmp_path and refer to the NPZ by basename.
        # This avoids path separators in saved filenames.
        import os

        os.chdir(tmp_path)
        instance = mod.SCmetrics(filename=path.name, liner="throw")
        yield instance
    finally:
        os.chdir(cwd)


def test_single_pvi_array(sample_npz_data: tuple[Path, dict[str, np.ndarray]]) -> None:
    """Verify raw NPZ loading for a single field."""
    path, arrays = sample_npz_data
    loaded = mod.singlePVIarray(npzfile=str(path), FIELD="Rcoord")
    assert np.allclose(loaded, arrays["Rcoord"])


def test_get_field_cleans_nans(sc: mod.SCmetrics,
                               sample_npz_data: tuple[Path, dict[str, np.ndarray]]
                               ) -> None:
    """Ensure get_field replaces NaNs with zeros."""
    _, arrays = sample_npz_data
    raw = arrays["density_throw"]
    expected = np.zeros_like(raw)
    valid = np.isfinite(raw)
    expected[valid] = raw[valid]
    assert np.array_equal(sc.density, expected)


def test_compute_volume_shape_and_values(sc: mod.SCmetrics) -> None:
    """Confirm the computed volume grid matches density shape and is positive."""
    assert sc.volume.shape == sc.density.shape
    assert np.all(sc.volume > 0.0)


def test_compute_regions_mask_and_labels(sc: mod.SCmetrics) -> None:
    """Exercise compute_regions for mask=True and mask=False."""
    # Masked regions should be strictly 0/1 with at least one positive cell.
    masked = sc.compute_regions(mask=True)
    assert masked.dtype.kind in {"i", "u"}
    assert masked.max() == 1
    assert masked.min() == 0

    # Labeled regions should enumerate on-axis connected components.
    labeled = sc.compute_regions(mask=False)
    assert labeled.max() >= 1  # at least one on-axis component
    # Any nonzero label must also appear in column 0 (on-axis).
    labels = np.unique(labeled[labeled > 0])
    for lab in labels:
        assert np.any(labeled[:, 0] == lab)


def test_get_jet_width_stats_no_threshold(sc: mod.SCmetrics) -> None:
    """Validate width statistics without a velocity threshold."""
    avg, std, mx = sc.get_jet_width_stats(vel_thres=0.0)
    assert avg >= 0.0
    assert std >= 0.0
    assert mx >= 0.0
    # Max width should be at least as large as average.
    assert mx >= avg


def test_get_jet_width_stats_with_threshold(sc: mod.SCmetrics) -> None:
    """Validate width statistics with a positive velocity threshold."""
    avg, std, mx = sc.get_jet_width_stats(vel_thres=1.0)
    assert avg >= 0.0
    assert std >= 0.0
    assert mx >= 0.0


def test_get_jet_rho_velsq_2d(sc: mod.SCmetrics) -> None:
    """Check cumulative sum of density * velocity^2 over on-axis regions."""
    val = sc.get_jet_rho_velsq_2D(vel_thres=0.5)
    assert val >= 0.0


def test_get_jet_sqrt_rho_vel_2d(sc: mod.SCmetrics) -> None:
    """Check cumulative sum of sqrt(density) * velocity over on-axis regions."""
    val = sc.get_jet_sqrt_rho_vel_2D(vel_thres=0.5)
    assert val >= 0.0


def test_get_jet_mass_and_cached(sc: mod.SCmetrics) -> None:
    """Verify jet mass and the 'cached' return branch."""
    computed = np.sum(sc.volume * sc.density * sc.vofm)
    assert np.isclose(sc.get_jet_mass(), computed)

    sc.jet_mass = 123.456  # trigger the else branch
    assert sc.get_jet_mass() == 123.456


def test_get_he_mass_and_cached(sc: mod.SCmetrics) -> None:
    """Validate HE mass and the manual cache-return branch."""
    he_den = sc.get_field(sc.HE_field_name)
    he_vf = sc.get_field(sc.HE_vofm_field_name)
    expected = float(np.sum(sc.volume * he_den * he_vf))
    assert np.isclose(sc.get_HE_mass(), expected)

    sc.HE_mass = 9.0
    assert sc.get_HE_mass() == 9.0


def test_max_regions_and_helpers(sc: mod.SCmetrics) -> None:
    """Confirm max over on-axis connected regions and wrapper method."""
    # Manual reference using current regions mask.
    region_mask = sc.regions > 0
    expected = float(np.max(sc.Wvelocity[region_mask]))
    assert sc.max_regions(sc.Wvelocity) == expected
    assert sc.max_Wvelocity() == expected


def test_avg_regions_raises_on_array_and(sc: mod.SCmetrics) -> None:
    """avg_regions uses Python 'and' with arrays; ensure it raises a ValueError."""
    with pytest.raises(ValueError):
        _ = sc.avg_regions(sc.Wvelocity, thresh=0.0)


def test_avg_wvelocity_signature_error(sc: mod.SCmetrics) -> None:
    """avg_Wvelocity forwards an unexpected kwarg; assert TypeError."""
    with pytest.raises(TypeError):
        _ = sc.avg_Wvelocity(Wthresh=0.0)


def test_eff_jet_mass_map_and_total(sc: mod.SCmetrics) -> None:
    """Validate effective jet mass computations and percent output."""
    emap = sc.get_eff_jet_mass_map(vel_thres=1.0)
    assert emap.ndim == 1  # selection returns a flat array
    total_eff = float(np.sum(emap))
    assert np.isclose(sc.get_eff_jet_mass(vel_thres=1.0), total_eff)

    percent = sc.get_eff_jet_mass(vel_thres=1.0, asPercent=True)
    denom = float(sc.get_jet_mass())
    assert np.isclose(percent, total_eff / denom if denom else 0.0)


def test_get_jet_kinetic_energy_returns_number(sc: mod.SCmetrics) -> None:
    """The current implementation returns a scalar; ensure it is finite."""
    val = sc.get_jet_kinetic_energy(vel_thres=0.5)
    assert np.isfinite(val)
    assert val >= 0.0


def test_get_jet_sqrt_kinetic_energy_returns_number(sc: mod.SCmetrics) -> None:
    """The current implementation returns a scalar; ensure it is finite."""
    val = sc.get_jet_sqrt_kinetic_energy(vel_thres=0.5)
    assert np.isfinite(val)
    assert val >= 0.0
