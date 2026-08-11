"""Tests for rotary positional embedding primitives.

This module contains unit tests for the self-contained (``diffusers``-free)
transformer primitives used by the plain-ViT backbone:

- SwiGLU: gated-linear MLP activation.
- get_1d_rotary_pos_embed: 1-D rotary cos/sin generation.
- apply_rotary_emb: rotary application to queries/keys.
- RotaryPositionalEmbeddingFromCenters: continuous multi-axis RoPE.
- make_regular_centers: regular-grid patch-center generation.

Tests cover initialization, forward passes, shape validation, mathematical
properties (norm preservation, relative-position invariance), and error
handling for each primitive.
"""

import math

import pytest
import torch
import torch.nn as nn

from yoke.models.vit.rope import (
    RotaryPositionalEmbeddingFromCenters,
    SwiGLU,
    apply_rotary_emb,
    get_1d_rotary_pos_embed,
    make_regular_centers,
)


@pytest.fixture
def device() -> str:
    """Get device for testing.

    Returns:
        Device string ('cuda' or 'cpu').
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# Tests for SwiGLU
# ============================================================================


def test_swiglu_init() -> None:
    """Test SwiGLU initialization."""
    swiglu = SwiGLU(dim_in=32, dim_out=64)

    assert isinstance(swiglu, SwiGLU)
    assert isinstance(swiglu.proj, nn.Linear)
    assert swiglu.proj.in_features == 32
    # Internal projection maps to 2 * dim_out for the gate split.
    assert swiglu.proj.out_features == 128
    assert isinstance(swiglu.activation, nn.SiLU)


def test_swiglu_init_no_bias() -> None:
    """Test SwiGLU initialization without bias."""
    swiglu = SwiGLU(dim_in=32, dim_out=64, bias=False)

    assert swiglu.proj.bias is None


def test_swiglu_forward_shape(device: str) -> None:
    """Test SwiGLU forward pass output shape.

    Args:
        device: Device for computation.
    """
    swiglu = SwiGLU(dim_in=32, dim_out=64).to(device)
    x = torch.randn(3, 10, 32, device=device)

    output = swiglu(x)

    assert output.shape == (3, 10, 64)
    assert not torch.isnan(output).any()


def test_swiglu_forward_values(device: str) -> None:
    """Test SwiGLU forward computes a * SiLU(b).

    Args:
        device: Device for computation.
    """
    swiglu = SwiGLU(dim_in=8, dim_out=4).to(device)
    x = torch.randn(2, 5, 8, device=device)

    proj = swiglu.proj(x)
    a, b = proj.chunk(2, dim=-1)
    expected = a * torch.nn.functional.silu(b)

    output = swiglu(x)

    assert torch.allclose(output, expected, atol=1e-6)


# ============================================================================
# Tests for get_1d_rotary_pos_embed
# ============================================================================


def test_get_1d_rotary_pos_embed_shape(device: str) -> None:
    """Test 1-D rotary embedding output shape.

    Args:
        device: Device for computation.
    """
    dim = 16
    pos = torch.arange(10, device=device).float()

    cos, sin = get_1d_rotary_pos_embed(dim, pos)

    assert cos.shape == (10, dim)
    assert sin.shape == (10, dim)


def test_get_1d_rotary_pos_embed_odd_dim_raises(device: str) -> None:
    """Test 1-D rotary embedding requires even dimension.

    Args:
        device: Device for computation.
    """
    pos = torch.arange(10, device=device).float()

    with pytest.raises(AssertionError):
        get_1d_rotary_pos_embed(15, pos)


def test_get_1d_rotary_pos_embed_value_range(device: str) -> None:
    """Test 1-D rotary embedding values lie in [-1, 1].

    Args:
        device: Device for computation.
    """
    pos = torch.linspace(0, 100, 20, device=device)

    cos, sin = get_1d_rotary_pos_embed(32, pos)

    assert torch.all(cos >= -1.0) and torch.all(cos <= 1.0)
    assert torch.all(sin >= -1.0) and torch.all(sin <= 1.0)


def test_get_1d_rotary_pos_embed_repeat_interleave(device: str) -> None:
    """Test consecutive feature pairs share the same rotary angle.

    Args:
        device: Device for computation.
    """
    dim = 8
    pos = torch.tensor([3.0], device=device)

    cos, sin = get_1d_rotary_pos_embed(dim, pos)

    # repeat_interleave(2) means index 0==1, 2==3, etc.
    for i in range(0, dim, 2):
        assert torch.allclose(cos[0, i], cos[0, i + 1])
        assert torch.allclose(sin[0, i], sin[0, i + 1])


def test_get_1d_rotary_pos_embed_zero_position(device: str) -> None:
    """Test rotary embedding at position zero is cos=1, sin=0.

    Args:
        device: Device for computation.
    """
    cos, sin = get_1d_rotary_pos_embed(16, torch.zeros(1, device=device))

    assert torch.allclose(cos, torch.ones_like(cos))
    assert torch.allclose(sin, torch.zeros_like(sin))


def test_get_1d_rotary_pos_embed_ntk_factor(device: str) -> None:
    """Test ntk_factor changes the resulting frequencies.

    Args:
        device: Device for computation.
    """
    pos = torch.linspace(0, 10, 5, device=device)

    cos1, _ = get_1d_rotary_pos_embed(16, pos, ntk_factor=1.0)
    cos2, _ = get_1d_rotary_pos_embed(16, pos, ntk_factor=2.0)

    assert not torch.allclose(cos1, cos2)


# ============================================================================
# Tests for apply_rotary_emb
# ============================================================================


def test_apply_rotary_emb_shape(device: str) -> None:
    """Test apply_rotary_emb preserves tensor shape.

    Args:
        device: Device for computation.
    """
    dim = 16
    N = 10
    pos = torch.arange(N, device=device).float()
    cos, sin = get_1d_rotary_pos_embed(dim, pos)

    x = torch.randn(2, N, 4, dim, device=device)
    out = apply_rotary_emb(x, (cos, sin), sequence_dim=1)

    assert out.shape == x.shape


def test_apply_rotary_emb_sequence_dim_2(device: str) -> None:
    """Test apply_rotary_emb with sequence_dim=2 layout.

    Args:
        device: Device for computation.
    """
    dim = 16
    N = 10
    pos = torch.arange(N, device=device).float()
    cos, sin = get_1d_rotary_pos_embed(dim, pos)

    # (B, H, N, D_head)
    x = torch.randn(2, 4, N, dim, device=device)
    out = apply_rotary_emb(x, (cos, sin), sequence_dim=2)

    assert out.shape == x.shape


def test_apply_rotary_emb_invalid_sequence_dim_raises(device: str) -> None:
    """Test apply_rotary_emb rejects unsupported sequence_dim.

    Args:
        device: Device for computation.
    """
    dim = 16
    N = 4
    pos = torch.arange(N, device=device).float()
    cos, sin = get_1d_rotary_pos_embed(dim, pos)
    x = torch.randn(2, N, 4, dim, device=device)

    with pytest.raises(ValueError, match="Unsupported sequence_dim"):
        apply_rotary_emb(x, (cos, sin), sequence_dim=3)


def test_apply_rotary_emb_zero_position_identity(device: str) -> None:
    """Test rotation at position zero is the identity.

    Args:
        device: Device for computation.
    """
    dim = 16
    cos, sin = get_1d_rotary_pos_embed(dim, torch.zeros(1, device=device))
    x = torch.randn(2, 1, 4, dim, device=device)

    out = apply_rotary_emb(x, (cos, sin), sequence_dim=1)

    assert torch.allclose(out, x, atol=1e-6)


def test_apply_rotary_emb_norm_preserving(device: str) -> None:
    """Test rotary embedding preserves per-token vector norm.

    Args:
        device: Device for computation.
    """
    dim = 16
    N = 6
    pos = torch.arange(N, device=device).float()
    cos, sin = get_1d_rotary_pos_embed(dim, pos)
    x = torch.randn(2, N, 4, dim, device=device)

    out = apply_rotary_emb(x, (cos, sin), sequence_dim=1)

    in_norm = x.norm(dim=-1)
    out_norm = out.norm(dim=-1)
    assert torch.allclose(in_norm, out_norm, atol=1e-5)


def test_apply_rotary_emb_relative_position(device: str) -> None:
    """Test rotary dot product depends only on relative position.

    Args:
        device: Device for computation.
    """
    dim = 32
    # Two absolute positions with the same offset should produce equal
    # query-key dot products after rotation.
    q = torch.randn(1, 1, 1, dim, device=device)
    k = torch.randn(1, 1, 1, dim, device=device)

    def rotated_dot(p_q: float, p_k: float) -> torch.Tensor:
        cos_q, sin_q = get_1d_rotary_pos_embed(
            dim, torch.tensor([p_q], device=device)
        )
        cos_k, sin_k = get_1d_rotary_pos_embed(
            dim, torch.tensor([p_k], device=device)
        )
        q_rot = apply_rotary_emb(q, (cos_q, sin_q), sequence_dim=1)
        k_rot = apply_rotary_emb(k, (cos_k, sin_k), sequence_dim=1)
        return (q_rot * k_rot).sum()

    dot_a = rotated_dot(2.0, 5.0)  # offset -3
    dot_b = rotated_dot(10.0, 13.0)  # offset -3
    assert torch.allclose(dot_a, dot_b, atol=1e-4)


# ============================================================================
# Tests for RotaryPositionalEmbeddingFromCenters
# ============================================================================


def test_rope_from_centers_init_scalar_theta() -> None:
    """Test RoPE-from-centers broadcasts a scalar theta over axes."""
    rope = RotaryPositionalEmbeddingFromCenters(rope_dim_list=[8, 8], theta=10000.0)

    assert rope.num_segments == 2
    assert rope.theta == [10000.0, 10000.0]


def test_rope_from_centers_init_list_theta() -> None:
    """Test RoPE-from-centers accepts a per-axis theta list."""
    rope = RotaryPositionalEmbeddingFromCenters(
        rope_dim_list=[8, 8], theta=[100.0, 200.0]
    )

    assert rope.theta == [100.0, 200.0]


def test_rope_from_centers_init_theta_length_mismatch() -> None:
    """Test RoPE-from-centers rejects a mismatched theta length."""
    with pytest.raises(AssertionError):
        RotaryPositionalEmbeddingFromCenters(
            rope_dim_list=[8, 8], theta=[100.0, 200.0, 300.0]
        )


def test_rope_from_centers_init_scale_none() -> None:
    """Test RoPE-from-centers defaults to unit scaling when scale is None."""
    rope = RotaryPositionalEmbeddingFromCenters(rope_dim_list=[8, 8], scale=None)

    assert torch.allclose(rope.scale_tensor, torch.ones(2))


def test_rope_from_centers_init_scale_scalar() -> None:
    """Test RoPE-from-centers broadcasts a scalar scale over axes."""
    rope = RotaryPositionalEmbeddingFromCenters(rope_dim_list=[8, 8], scale=3.0)

    assert torch.allclose(rope.scale_tensor, torch.tensor([3.0, 3.0]))


def test_rope_from_centers_init_scale_list() -> None:
    """Test RoPE-from-centers accepts a per-axis scale list."""
    rope = RotaryPositionalEmbeddingFromCenters(
        rope_dim_list=[8, 8], scale=[2.0, 5.0]
    )

    assert torch.allclose(rope.scale_tensor, torch.tensor([2.0, 5.0]))


def test_rope_from_centers_init_scale_length_mismatch() -> None:
    """Test RoPE-from-centers rejects a mismatched scale length."""
    with pytest.raises(AssertionError):
        RotaryPositionalEmbeddingFromCenters(
            rope_dim_list=[8, 8], scale=[2.0, 5.0, 7.0]
        )


def test_rope_from_centers_forward_2d_shape(device: str) -> None:
    """Test RoPE-from-centers forward with 2-D centers input.

    Args:
        device: Device for computation.
    """
    head_dim = 32
    rope = RotaryPositionalEmbeddingFromCenters(
        rope_dim_list=[head_dim // 2, head_dim // 2]
    ).to(device)
    centers = torch.rand(50, 2, device=device)

    cos, sin = rope(centers)

    assert cos.shape == (50, head_dim)
    assert sin.shape == (50, head_dim)


def test_rope_from_centers_forward_3d_shape(device: str) -> None:
    """Test RoPE-from-centers flattens a batched centers input.

    Args:
        device: Device for computation.
    """
    head_dim = 32
    rope = RotaryPositionalEmbeddingFromCenters(
        rope_dim_list=[head_dim // 2, head_dim // 2]
    ).to(device)
    # (B, N, D) with B=2, N=25 -> flattened to 50.
    centers = torch.rand(2, 25, 2, device=device)

    cos, sin = rope(centers)

    assert cos.shape == (50, head_dim)
    assert sin.shape == (50, head_dim)


def test_rope_from_centers_forward_bad_ndim_raises(device: str) -> None:
    """Test RoPE-from-centers rejects centers with wrong ndim.

    Args:
        device: Device for computation.
    """
    rope = RotaryPositionalEmbeddingFromCenters(rope_dim_list=[8, 8]).to(device)
    centers = torch.rand(2, 3, 4, 2, device=device)

    with pytest.raises(ValueError, match=r"centers must be"):
        rope(centers)


def test_rope_from_centers_forward_bad_last_dim_raises(device: str) -> None:
    """Test RoPE-from-centers rejects centers whose last dim != num axes.

    Args:
        device: Device for computation.
    """
    rope = RotaryPositionalEmbeddingFromCenters(rope_dim_list=[8, 8]).to(device)
    centers = torch.rand(50, 3, device=device)

    with pytest.raises(ValueError, match=r"Last dimension of centers"):
        rope(centers)


def test_rope_from_centers_three_axis(device: str) -> None:
    """Test RoPE-from-centers supports more than two axes.

    Args:
        device: Device for computation.
    """
    rope = RotaryPositionalEmbeddingFromCenters(rope_dim_list=[8, 8, 16]).to(device)
    centers = torch.rand(30, 3, device=device)

    cos, sin = rope(centers)

    assert cos.shape == (30, 32)
    assert sin.shape == (30, 32)


# ============================================================================
# Tests for make_regular_centers
# ============================================================================


def test_make_regular_centers_shape(device: str) -> None:
    """Test make_regular_centers output shape.

    Args:
        device: Device for computation.
    """
    centers = make_regular_centers((1120, 800), (10, 10), device=device)

    assert centers.shape == (112 * 80, 2)


def test_make_regular_centers_range(device: str) -> None:
    """Test make_regular_centers values lie in (0, 1).

    Args:
        device: Device for computation.
    """
    centers = make_regular_centers((64, 32), (8, 8), device=device)

    assert torch.all(centers > 0.0)
    assert torch.all(centers < 1.0)


def test_make_regular_centers_values(device: str) -> None:
    """Test make_regular_centers computes the expected first/last center.

    Args:
        device: Device for computation.
    """
    H, W = 40, 20
    p_h, p_w = 10, 10
    centers = make_regular_centers((H, W), (p_h, p_w), device=device)

    # First patch center (row 0, col 0): [x, y].
    expected_first_x = 0.5 * p_w / W
    expected_first_y = 0.5 * p_h / H
    assert math.isclose(centers[0, 0].item(), expected_first_x, rel_tol=1e-6)
    assert math.isclose(centers[0, 1].item(), expected_first_y, rel_tol=1e-6)


def test_make_regular_centers_row_major_order(device: str) -> None:
    """Test centers are emitted in row-major (ij) patch order.

    Args:
        device: Device for computation.
    """
    H, W = 20, 30
    p = (10, 10)
    W_p = W // p[1]
    centers = make_regular_centers((H, W), p, device=device)

    # Within the first row the y-coordinate is constant while x increases.
    first_row = centers[:W_p]
    assert torch.allclose(first_row[:, 1], first_row[0, 1] * torch.ones(W_p))
    assert torch.all(first_row[1:, 0] > first_row[:-1, 0])


def test_make_regular_centers_default_device() -> None:
    """Test make_regular_centers defaults to CPU."""
    centers = make_regular_centers((32, 32), (8, 8))

    assert centers.device.type == "cpu"
