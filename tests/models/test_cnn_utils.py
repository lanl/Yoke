"""Test for `src/yoke/models/cnn_utils`."""

import pytest
from yoke.models.cnn_utils import conv2d_shape, convtranspose2d_shape


def test_conv2d_shape() -> None:
    """Tests the conv2d_shape method."""
    # Define test cases: (w, h, k, s_w, s_h, p_w, p_h,
    # expected_new_w, expected_new_h, expected_total)
    test_cases = [
        (100, 150, 3, 2, 1, 2, 1, 51, 150, 7650),
        (28, 28, 3, 1, 1, 1, 1, 28, 28, 784),
        (32, 32, 5, 1, 1, 0, 0, 28, 28, 784),
        (64, 64, 3, 2, 2, 1, 1, 32, 32, 1024),
    ]

    for (
        w,
        h,
        k,
        s_w,
        s_h,
        p_w,
        p_h,
        expected_new_w,
        expected_new_h,
        expected_total,
    ) in test_cases:
        new_w, new_h, total = conv2d_shape(w, h, k, s_w, s_h, p_w, p_h)
        assert new_w == expected_new_w, (
            f"Expected width {expected_new_w}, but got {new_w}"
        )
        assert new_h == expected_new_h, (
            f"Expected height {expected_new_h}, but got {new_h}"
        )
        assert total == expected_total, (
            f"Expected total {expected_total}, but got {total}"
        )


@pytest.mark.parametrize(
    "w, h, k_w, k_h, s_w, s_h, p_w, p_h, op_w, op_h, d_w, d_h, expected",
    [
        # Case 1: Basic dimensions, no padding, stride = 1
        (4, 4, 3, 3, 1, 1, 0, 0, 0, 0, 1, 1, (6, 6, 36)),
        # Case 2: Basic dimensions with padding and stride = 1
        (4, 4, 3, 3, 1, 1, 1, 1, 0, 0, 1, 1, (4, 4, 16)),
        # Case 3: Stride > 1, no padding
        (4, 4, 3, 3, 2, 2, 0, 0, 0, 0, 1, 1, (9, 9, 81)),
        # Case 4: Stride > 1 with padding
        (4, 4, 3, 3, 2, 2, 1, 1, 0, 0, 1, 1, (7, 7, 49)),
        # Case 5: Output padding applied
        (4, 4, 3, 3, 2, 2, 0, 0, 1, 1, 1, 1, (10, 10, 100)),
        # Case 6: Larger dilation applied
        (4, 4, 3, 3, 1, 1, 0, 0, 0, 0, 2, 2, (8, 8, 64)),
    ],
)
def test_convtranspose2d_shape(
    w: int,
    h: int,
    k_w: int,
    k_h: int,
    s_w: int,
    s_h: int,
    p_w: int,
    p_h: int,
    op_w: int,
    op_h: int,
    d_w: int,
    d_h: int,
    expected: tuple[int, int, int],
) -> None:
    """Test convtranspose2d_shape with various parameters."""
    result = convtranspose2d_shape(
        w, h, k_w, k_h, s_w, s_h, p_w, p_h, op_w, op_h, d_w, d_h
    )
    assert result == expected
