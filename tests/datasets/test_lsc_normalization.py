"""Unit tests for the *lsc_normalization* module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from yoke.datasets.lsc_normalization import (
    DEFAULT_CHANNEL_LIST,
    classify_channel,
    compute_channel_stats,
    load_channel_norm,
    normalize_field,
    save_channel_norm,
    signed_log1p,
)


def test_default_channel_list_is_the_8ch_set() -> None:
    """DEFAULT_CHANNEL_LIST matches the 8-channel density + velocity set."""
    assert DEFAULT_CHANNEL_LIST == [
        "density_case",
        "density_cushion",
        "density_maincharge",
        "density_outside_air",
        "density_striker",
        "density_throw",
        "Uvelocity",
        "Wvelocity",
    ]


@pytest.mark.parametrize(
    "channel,expected",
    [
        ("density_case", "density"),
        ("density_maincharge", "density"),
        ("Uvelocity", "velocity"),
        ("Wvelocity", "velocity"),
        ("pressure_case", "log_range"),
        ("energy_throw", "log_range"),
    ],
)
def test_classify_channel(channel: str, expected: str) -> None:
    """classify_channel routes known channel-name patterns correctly."""
    assert classify_channel(channel) == expected


def test_classify_channel_unknown_raises() -> None:
    """classify_channel raises ValueError for an unrecognized channel name."""
    with pytest.raises(ValueError, match="does not match a known"):
        classify_channel("bogus_channel")


def test_signed_log1p_preserves_zero_and_sign() -> None:
    """signed_log1p maps 0 -> 0 and preserves sign."""
    x = np.array([-5.0, 0.0, 5.0])
    y = signed_log1p(x)
    assert y[1] == 0.0
    assert y[0] < 0.0
    assert y[2] > 0.0
    assert np.allclose(y, np.sign(x) * np.log1p(np.abs(x)))


def test_compute_channel_stats_density_is_one_sided_and_zero_preserving() -> None:
    """Density stats use a one-sided percentile of nonzero magnitudes."""
    rng = np.random.default_rng(0)
    voxels = np.concatenate([np.zeros(1000), rng.uniform(0, 10, 1000)])
    record = compute_channel_stats("density_case", voxels, 99.9, 0.1, 99.9)

    assert record["family"] == "density"
    assert record["scale"] > 0.0

    normalized = normalize_field(np.array([0.0, 5.0]), record)
    assert normalized[0] == 0.0


def test_compute_channel_stats_density_no_nonzero_raises() -> None:
    """An all-zero density sample cannot yield a scale factor."""
    with pytest.raises(ValueError, match="No nonzero voxels"):
        compute_channel_stats("density_case", np.zeros(100), 99.9, 0.1, 99.9)


def test_compute_channel_stats_velocity_is_symmetric_and_zero_preserving() -> None:
    """Velocity stats use a symmetric percentile scale; zero maps to zero."""
    rng = np.random.default_rng(1)
    voxels = rng.normal(0.0, 3.0, 5000)
    record = compute_channel_stats("Uvelocity", voxels, 99.9, 0.1, 99.9)

    assert record["family"] == "velocity"
    assert record["scale"] > 0.0

    normalized = normalize_field(np.array([0.0]), record)
    assert normalized[0] == 0.0

    p_low = np.percentile(voxels, 0.1)
    p_high = np.percentile(voxels, 99.9)
    assert record["scale"] == pytest.approx(max(abs(p_low), abs(p_high)))


def test_compute_channel_stats_velocity_empty_raises() -> None:
    """An empty velocity sample raises ValueError."""
    with pytest.raises(ValueError, match="No voxels sampled"):
        compute_channel_stats("Uvelocity", np.array([]), 99.9, 0.1, 99.9)


def test_compute_channel_stats_log_range_is_signed_and_zero_preserving() -> None:
    """Log-range (pressure/energy) stats apply signed-log1p then a symmetric scale."""
    rng = np.random.default_rng(2)
    voxels = rng.normal(0.0, 1.0e6, 5000)  # pressure can be negative (stress trace)
    record = compute_channel_stats("pressure_maincharge", voxels, 99.9, 0.1, 99.9)

    assert record["family"] == "log_range"
    assert record["scale"] > 0.0

    normalized = normalize_field(np.array([0.0]), record)
    assert normalized[0] == 0.0

    # Sign is preserved through the signed-log1p transform.
    neg = normalize_field(np.array([-1000.0]), record)
    pos = normalize_field(np.array([1000.0]), record)
    assert neg[0] < 0.0
    assert pos[0] > 0.0


def test_compute_channel_stats_log_range_empty_raises() -> None:
    """An empty log-range sample raises ValueError."""
    with pytest.raises(ValueError, match="No voxels sampled"):
        compute_channel_stats("pressure_case", np.array([]), 99.9, 0.1, 99.9)


def test_compute_channel_stats_degenerate_scale_raises() -> None:
    """A degenerate (constant nonzero) density sample yields scale<=0 and raises."""
    # All identical nonzero values -> both percentiles equal that constant, so
    # the computed scale is positive here; construct a genuinely degenerate
    # case instead: nonzero values that straddle to a computed scale of 0 is
    # not reachable for abs-percentile of positive constants, so directly
    # drive the velocity branch to a zero scale using all-zero voxels (which
    # the "empty" check does not catch since the array is non-empty).
    voxels = np.zeros(500)
    with pytest.raises(ValueError, match="non-positive scale"):
        compute_channel_stats("Uvelocity", voxels, 99.9, 0.1, 99.9)


def test_compute_channel_stats_filters_non_finite() -> None:
    """Non-finite voxel values (NaN/Inf) are filtered before stats computation."""
    rng = np.random.default_rng(3)
    clean = rng.uniform(0.0, 10.0, 2000)
    with_nans = np.concatenate([clean, [np.nan, np.inf, -np.inf]])
    record_clean = compute_channel_stats("density_case", clean, 99.9, 0.1, 99.9)
    record_with_nans = compute_channel_stats("density_case", with_nans, 99.9, 0.1, 99.9)
    assert record_clean["scale"] == pytest.approx(record_with_nans["scale"])


def test_save_and_load_channel_norm_round_trip() -> None:
    """save_channel_norm/load_channel_norm round-trip exactly."""
    stats = {
        "density_case": {"family": "density", "scale": 1.234},
        "Uvelocity": {"family": "velocity", "scale": 5.678},
        "pressure_case": {"family": "log_range", "scale": 9.101},
    }
    with tempfile.TemporaryDirectory() as tmp:
        fileout = str(Path(tmp) / "norm.npz")
        save_channel_norm(fileout, stats)
        loaded = load_channel_norm(fileout)

    assert loaded.keys() == stats.keys()
    for channel, record in stats.items():
        assert loaded[channel]["family"] == record["family"]
        assert loaded[channel]["scale"] == pytest.approx(record["scale"])


def test_normalize_field_density_and_velocity_are_pure_scale() -> None:
    """normalize_field divides by scale for density/velocity (no log, no shift)."""
    record = {"family": "density", "scale": 2.0}
    raw = np.array([0.0, 1.0, 4.0])
    assert np.allclose(normalize_field(raw, record), raw / 2.0)

    record_v = {"family": "velocity", "scale": 4.0}
    raw_v = np.array([-4.0, 0.0, 4.0])
    assert np.allclose(normalize_field(raw_v, record_v), raw_v / 4.0)


def test_normalize_field_log_range_applies_signed_log1p_then_scale() -> None:
    """normalize_field applies signed-log1p before scaling for log_range fields."""
    record = {"family": "log_range", "scale": 2.0}
    raw = np.array([-10.0, 0.0, 10.0])
    expected = signed_log1p(raw) / 2.0
    assert np.allclose(normalize_field(raw, record), expected)
