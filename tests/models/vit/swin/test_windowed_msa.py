"""Tests for try/except validations in windowed MSA classes.

This module focuses exclusively on the assertion-based validation wrapped in
try/except blocks within four classes: WindowMSA, ShiftedWindowMSA,
WindowCosMSA, and ShiftedWindowCosMSA.
"""

from __future__ import annotations

import pytest

from yoke.models.vit.swin.windowed_msa import (
    WindowMSA,
    ShiftedWindowMSA,
    WindowCosMSA,
    ShiftedWindowCosMSA,
    )


ValidParams = tuple[int, int, tuple[int, int], tuple[int, int]]


def _assert_label_next(args: tuple[object, ...], label: str, value: object) -> None:
    """Assert that ``label`` appear in ``args`` and is immediately followed by ``value``.

    Args:
        args: The exception ``.args`` tuple to scan.
        label: The label string expected in ``args``.
        value: The value expected immediately after ``label``.
    """
    arr = list(args)
    assert label in arr
    assert arr[arr.index(label) + 1] == value


def _valid_params() -> ValidParams:
    """Return a set of parameters that satisfy all class preconditions.

    Returns:
        A tuple of (emb_size, num_heads, patch_grid_size, window_size) that
        divides cleanly for the base and shifted variants.
    """
    return 64, 8, (16, 32), (8, 4)


@pytest.mark.parametrize(
    "cls",
    [WindowMSA, ShiftedWindowMSA, WindowCosMSA, ShiftedWindowCosMSA],
)
def test_valid_instantiation_no_assert(cls: type[object]) -> None:
    """Instantiate each class with valid params and ensure no assertion raises.

    Args:
        cls: The attention class to instantiate.
    """
    emb, heads, grid, win = _valid_params()
    _ = cls(emb_size=emb, num_heads=heads, patch_grid_size=grid, window_size=win)


@pytest.mark.parametrize(
    "cls",
    [WindowMSA, ShiftedWindowMSA, WindowCosMSA, ShiftedWindowCosMSA],
)
def test_embed_not_divisible_raises(cls: type[object]) -> None:
    """Embedding size not divisible by heads should raise with augmented args.

    Args:
        cls: The attention class to instantiate.
    """
    emb, heads, grid, win = _valid_params()
    bad_emb = emb - 4  # 60 % 8 != 0
    with pytest.raises(AssertionError) as ei:
        _ = cls(
            emb_size=bad_emb, num_heads=heads, patch_grid_size=grid, window_size=win
        )
    # The first arg is the message; additional context is appended.
    args = ei.value.args
    msg = "Embedding size not divisible by number of heads"
    assert msg in str(ei.value)
    _assert_label_next(args, "Embedding size:", bad_emb)
    _assert_label_next(args, "Number of heads:", heads)


@pytest.mark.parametrize(
    "cls",
    [WindowMSA, ShiftedWindowMSA, WindowCosMSA, ShiftedWindowCosMSA],
)
def test_patch_grid_height_not_divisible_raises(cls: type[object]) -> None:
    """Nondivisible patch-grid height should raise with helpful context.

    Args:
        cls: The attention class to instantiate.
    """
    emb, heads, grid, win = _valid_params()
    bad_grid = (grid[0] - 2, grid[1])  # 14 % 8 != 0
    with pytest.raises(AssertionError) as ei:
        _ = cls(
            emb_size=emb, num_heads=heads, patch_grid_size=bad_grid, window_size=win
        )
    args = ei.value.args
    assert "Patch-grid not divisible by window-size" in str(ei.value)
    _assert_label_next(args, "Patch-grid 1:", bad_grid[0])
    _assert_label_next(args, "Window-size 1:", win[0])


@pytest.mark.parametrize(
    "cls",
    [WindowMSA, ShiftedWindowMSA, WindowCosMSA, ShiftedWindowCosMSA],
)
def test_patch_grid_width_not_divisible_raises(cls: type[object]) -> None:
    """Nondivisible patch-grid width should raise with helpful context.

    Args:
        cls: The attention class to instantiate.
    """
    emb, heads, grid, win = _valid_params()
    bad_grid = (grid[0], grid[1] - 2)  # 30 % 4 != 0
    with pytest.raises(AssertionError) as ei:
        _ = cls(
            emb_size=emb, num_heads=heads, patch_grid_size=bad_grid, window_size=win
        )
    args = ei.value.args
    assert "Patch-grid not divisible by window-size" in str(ei.value)
    _assert_label_next(args, "Patch-grid 2:", bad_grid[1])
    _assert_label_next(args, "Window-size 2:", win[1])


def test_shifted_window_height_even_raises() -> None:
    """ShiftedWindowMSA requires even window height; odd should raise.

    Uses a patch grid divisible by the odd height so earlier checks pass and
    the test exclusively exercises the evenness assertion.
    """
    emb, heads, _, _ = _valid_params()
    grid = (14, 32)
    win = (7, 4)  # odd height; divisible with grid; should trip evenness check
    with pytest.raises(AssertionError) as ei:
        _ = ShiftedWindowMSA(
            emb_size=emb, num_heads=heads, patch_grid_size=grid, window_size=win
        )
    args = ei.value.args
    assert "Window height not divisble by 2" in str(ei.value)
    _assert_label_next(args, "Window height:", win[0])


def test_shifted_window_width_even_raises() -> None:
    """ShiftedWindowMSA requires even window width; odd should raise.

    Uses a patch grid divisible by the odd width so earlier checks pass and
    the test exclusively exercises the evenness assertion.
    """
    emb, heads, _, _ = _valid_params()
    grid = (16, 30)
    win = (8, 5)  # odd width; divisible with grid; should trip evenness check
    with pytest.raises(AssertionError) as ei:
        _ = ShiftedWindowMSA(
            emb_size=emb, num_heads=heads, patch_grid_size=grid, window_size=win
        )
    args = ei.value.args
    assert "Window width not divisble by 2" in str(ei.value)
    _assert_label_next(args, "Window width:", win[1])
