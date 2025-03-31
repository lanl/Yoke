"""Test training design helpers."""

import numpy as np

from yoke.helpers.training_design import (
    choose_downsample_factor,
    find_valid_pad,
    validate_patch_and_window,
)


def test_validate_patch_and_window() -> None:
    """Ensure model input shape validation helper works."""
    image_size = (1120, 400)
    patch_size = 5
    window_sizes = [(2, 2) for _ in range(4)]
    patch_merge_scales = [(2, 2) for _ in range(3)]
    valid = validate_patch_and_window(
        image_size=image_size,
        patch_size=patch_size,
        window_sizes=window_sizes,
        patch_merge_scales=patch_merge_scales,
    )
    assert np.all(valid), "validate_patch_and_window() failing for valid shapes!"

    invalid = validate_patch_and_window(
        image_size=image_size + np.array([1, 0]),
        patch_size=patch_size,
        window_sizes=window_sizes,
        patch_merge_scales=patch_merge_scales,
    )
    assert not np.all(invalid), (
        "validate_patch_and_window() failing to detect invalid shapes!"
    )


def test_find_valid_pad() -> None:
    """Ensure pad search can find a valid pad."""
    image_size = (1120 // 4, 400 // 4)
    patch_size = 5
    window_sizes = [(2, 2) for _ in range(4)]
    patch_merge_scales = [(2, 2) for _ in range(3)]
    pad_options = np.arange(100)
    pad_dim0, pad_dim1 = find_valid_pad(
        image_size=image_size,
        patch_size=patch_size,
        window_sizes=window_sizes,
        patch_merge_scales=patch_merge_scales,
        pad_options=pad_options,
    )
    assert (len(pad_dim0) > 0) and (len(pad_dim1) > 0), (
        "find_valid_pad() unable to find usable pad!"
    )


def test_choose_downsample_factor() -> None:
    """Ensure choose_downsample_factor() runs without crashing."""
    image_size = np.array([1120, 400])
    patch_size = 5
    window_sizes = [(2, 2) for _ in range(4)]
    patch_merge_scales = [(2, 2) for _ in range(3)]
    pad_options = np.arange(100)
    desired_scale_factor = 0.25
    max_scale_dev = 0.5
    choose_downsample_factor(
        image_size=image_size,
        patch_size=patch_size,
        window_sizes=window_sizes,
        patch_merge_scales=patch_merge_scales,
        pad_options=pad_options,
        desired_scale_factor=desired_scale_factor,
        max_scale_dev=max_scale_dev,
    )
