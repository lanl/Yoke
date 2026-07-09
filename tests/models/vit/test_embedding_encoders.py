"""Tests for embedding encoders used in ViT networks.

This module contains comprehensive unit tests for various embedding classes used
in Vision Transformer (ViT) architectures, including:

- VarEmbed: Variable encoding/embedding for tracking variables through layers
- PosEmbed: Position encoding/embedding for spatial awareness
- RelativePositionEmbed: Relative spatial encoding for SWIN attention
- TimeEmbed: Temporal encoding for lead time conditioning
- DiffusionTimeEmbed: Diffusion time encoding for score-based models

Tests cover initialization, forward passes, weight initialization, shape
validation, and edge cases for each embedding type.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from yoke.models.vit.embedding_encoders import (
    DiffusionTimeEmbed,
    PosEmbed,
    RelativePositionEmbed,
    TimeEmbed,
    VarEmbed,
    get_1d_sincos_pos_embed_from_grid,
    get_2d_sincos_pos_embed,
    get_2d_sincos_pos_embed_from_grid,
)


@pytest.fixture
def default_vars() -> list[str]:
    """Create default variable list fixture.

    Returns:
        List of variable names for testing.
    """
    return [
        "cu_pressure",
        "cu_density",
        "cu_temperature",
        "ss_pressure",
        "ss_density",
        "ss_temperature",
        "r_vel",
        "z_vel",
    ]


@pytest.fixture
def device() -> str:
    """Get device for testing.

    Returns:
        Device string ('cuda' or 'cpu').
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# Tests for sincos position embedding utility functions
# ============================================================================


def test_get_1d_sincos_pos_embed_from_grid_shape() -> None:
    """Test 1D sincos position embedding output shape."""
    embed_dim = 64
    num_positions = 10
    position = np.arange(num_positions).reshape(-1, 1)

    emb = get_1d_sincos_pos_embed_from_grid(embed_dim, position)

    assert emb.shape == (num_positions, embed_dim)


def test_get_1d_sincos_pos_embed_from_grid_values() -> None:
    """Test 1D sincos position embedding value ranges."""
    embed_dim = 64
    position = np.arange(10).reshape(-1, 1)

    emb = get_1d_sincos_pos_embed_from_grid(embed_dim, position)

    # Values should be in [-1, 1] range for sin/cos
    assert np.all(emb >= -1.0)
    assert np.all(emb <= 1.0)


def test_get_1d_sincos_pos_embed_from_grid_even_dim() -> None:
    """Test 1D sincos embedding requires even dimension."""
    embed_dim = 63  # Odd dimension
    position = np.arange(10).reshape(-1, 1)

    with pytest.raises(AssertionError):
        get_1d_sincos_pos_embed_from_grid(embed_dim, position)


def test_get_2d_sincos_pos_embed_from_grid_shape() -> None:
    """Test 2D sincos position embedding from grid output shape."""
    embed_dim = 64
    grid_h, grid_w = 8, 10
    grid = np.stack(np.meshgrid(np.arange(grid_w), np.arange(grid_h)), axis=0)
    grid = grid.reshape([2, 1, grid_h, grid_w])

    emb = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

    assert emb.shape == (grid_h * grid_w, embed_dim)


def test_get_2d_sincos_pos_embed_from_grid_even_dim() -> None:
    """Test 2D sincos embedding from grid requires even dimension."""
    embed_dim = 63  # Odd dimension
    grid = np.random.randn(2, 1, 8, 8)

    with pytest.raises(AssertionError):
        get_2d_sincos_pos_embed_from_grid(embed_dim, grid)


def test_get_2d_sincos_pos_embed_shape() -> None:
    """Test 2D sincos position embedding output shape."""
    embed_dim = 64
    grid_size_h, grid_size_w = 8, 10

    pos_embed = get_2d_sincos_pos_embed(
        embed_dim, grid_size_h, grid_size_w, cls_token=False
    )

    assert pos_embed.shape == (grid_size_h * grid_size_w, embed_dim)


def test_get_2d_sincos_pos_embed_with_cls_token() -> None:
    """Test 2D sincos position embedding with class token."""
    embed_dim = 64
    grid_size_h, grid_size_w = 8, 10

    pos_embed = get_2d_sincos_pos_embed(
        embed_dim, grid_size_h, grid_size_w, cls_token=True
    )

    # Should have one extra position for class token
    assert pos_embed.shape == (1 + grid_size_h * grid_size_w, embed_dim)
    # First row should be zeros for class token
    assert np.allclose(pos_embed[0], np.zeros(embed_dim))


def test_get_2d_sincos_pos_embed_even_dim() -> None:
    """Test 2D sincos embedding requires even dimension."""
    embed_dim = 63  # Odd dimension

    with pytest.raises(AssertionError):
        get_2d_sincos_pos_embed(embed_dim, 8, 8, cls_token=False)


# ============================================================================
# Tests for VarEmbed
# ============================================================================


def test_var_embed_init(default_vars: list[str]) -> None:
    """Test VarEmbed initialization.

    Args:
        default_vars: List of variable names.
    """
    embed_dim = 64
    var_embed = VarEmbed(default_vars, embed_dim)

    assert isinstance(var_embed, VarEmbed)
    assert var_embed.embed_dim == embed_dim
    assert len(var_embed.default_vars) == len(default_vars)
    assert var_embed.var_embed.shape == (1, len(default_vars), embed_dim)


def test_var_embed_var_map(default_vars: list[str]) -> None:
    """Test VarEmbed variable mapping creation.

    Args:
        default_vars: List of variable names.
    """
    embed_dim = 64
    var_embed = VarEmbed(default_vars, embed_dim)

    # Check that all variables are in the map
    for i, var in enumerate(default_vars):
        assert var in var_embed.var_map
        assert var_embed.var_map[var] == i


def test_var_embed_forward_shape(default_vars: list[str], device: str) -> None:
    """Test VarEmbed forward pass output shape.

    Args:
        default_vars: List of variable names.
        device: Device for computation.
    """
    embed_dim = 64
    batch_size = 3
    num_vars = 4
    num_tokens = 64

    var_embed = VarEmbed(default_vars, embed_dim).to(device)

    # Input shape: (B, V, L, D)
    x = torch.randn(batch_size, num_vars, num_tokens, embed_dim).to(device)
    in_vars = torch.tensor([0, 1, 2, 3]).to(device)

    output = var_embed(x, in_vars)

    assert output.shape == x.shape
    assert not torch.isnan(output).any()


def test_var_embed_forward_adds_embedding(default_vars: list[str], device: str) -> None:
    """Test VarEmbed forward pass adds embedding to input.

    Args:
        default_vars: List of variable names.
        device: Device for computation.
    """
    embed_dim = 64
    batch_size = 2
    num_vars = 3
    num_tokens = 32

    var_embed = VarEmbed(default_vars, embed_dim).to(device)

    x = torch.randn(batch_size, num_vars, num_tokens, embed_dim).to(device)
    in_vars = torch.tensor([0, 1, 2]).to(device)

    output = var_embed(x, in_vars)

    # Output should be different from input (embedding added)
    assert not torch.allclose(output, x)


def test_var_embed_weight_initialization(default_vars: list[str]) -> None:
    """Test VarEmbed weight initialization with sincos.

    Args:
        default_vars: List of variable names.
    """
    embed_dim = 64
    var_embed = VarEmbed(default_vars, embed_dim)

    # Check that weights are initialized (not all zeros)
    assert not torch.allclose(var_embed.var_embed, torch.zeros_like(var_embed.var_embed))


# ============================================================================
# Tests for PosEmbed
# ============================================================================


def test_pos_embed_init() -> None:
    """Test PosEmbed initialization."""
    embed_dim = 64
    patch_size = (16, 16)
    image_size = (128, 128)
    num_patches = 64

    pos_embed = PosEmbed(embed_dim, patch_size, image_size, num_patches)

    assert isinstance(pos_embed, PosEmbed)
    assert pos_embed.embed_dim == embed_dim
    assert pos_embed.patch_size == patch_size
    assert pos_embed.image_size == image_size
    assert pos_embed.num_patches == num_patches
    assert pos_embed.pos_embed.shape == (1, num_patches, embed_dim)


def test_pos_embed_forward_shape(device: str) -> None:
    """Test PosEmbed forward pass output shape.

    Args:
        device: Device for computation.
    """
    embed_dim = 64
    patch_size = (16, 16)
    image_size = (128, 128)
    num_patches = 64
    batch_size = 3

    pos_embed = PosEmbed(embed_dim, patch_size, image_size, num_patches).to(device)

    # Input shape: (B, L, D)
    x = torch.randn(batch_size, num_patches, embed_dim).to(device)

    output = pos_embed(x)

    assert output.shape == x.shape
    assert not torch.isnan(output).any()


def test_pos_embed_forward_adds_embedding(device: str) -> None:
    """Test PosEmbed forward pass adds embedding to input.

    Args:
        device: Device for computation.
    """
    embed_dim = 64
    patch_size = (16, 16)
    image_size = (128, 128)
    num_patches = 64
    batch_size = 2

    pos_embed = PosEmbed(embed_dim, patch_size, image_size, num_patches).to(device)

    x = torch.randn(batch_size, num_patches, embed_dim).to(device)

    output = pos_embed(x)

    # Output should be different from input (embedding added)
    assert not torch.allclose(output, x)


def test_pos_embed_weight_initialization() -> None:
    """Test PosEmbed weight initialization with sincos."""
    embed_dim = 64
    patch_size = (16, 16)
    image_size = (128, 128)
    num_patches = 64

    pos_embed = PosEmbed(embed_dim, patch_size, image_size, num_patches)

    # Check that weights are initialized (not all zeros)
    zeros_like_check = torch.zeros_like(pos_embed.pos_embed)
    assert not torch.allclose(pos_embed.pos_embed, zeros_like_check)


def test_pos_embed_different_image_sizes() -> None:
    """Test PosEmbed with different image and patch sizes."""
    embed_dim = 64
    patch_size = (10, 10)
    image_size = (1120, 800)
    num_patches = (1120 // 10) * (800 // 10)

    pos_embed = PosEmbed(embed_dim, patch_size, image_size, num_patches)

    assert pos_embed.pos_embed.shape == (1, num_patches, embed_dim)


# ============================================================================
# Tests for RelativePositionEmbed
# ============================================================================


def test_relative_position_embed_init() -> None:
    """Test RelativePositionEmbed initialization."""
    window_size = (8, 8)
    rel_pos_embed = RelativePositionEmbed(window_size)

    assert isinstance(rel_pos_embed, RelativePositionEmbed)
    assert rel_pos_embed.window_size == window_size
    expected_shape = (
        2 * window_size[0] - 1,
        2 * window_size[1] - 1,
    )
    assert rel_pos_embed.pos_embeddings.shape == expected_shape


def test_relative_position_embed_indices() -> None:
    """Test RelativePositionEmbed relative indices computation."""
    window_size = (4, 4)
    rel_pos_embed = RelativePositionEmbed(window_size)

    wh, ww = window_size
    expected_indices_shape = (wh * ww, wh * ww, 2)
    assert rel_pos_embed.relative_indices.shape == expected_indices_shape

    # Check index ranges after shifting
    assert rel_pos_embed.relative_indices[:, :, 0].min() == 0
    assert rel_pos_embed.relative_indices[:, :, 0].max() == 2 * wh - 2
    assert rel_pos_embed.relative_indices[:, :, 1].min() == 0
    assert rel_pos_embed.relative_indices[:, :, 1].max() == 2 * ww - 2


def test_relative_position_embed_forward_shape(device: str) -> None:
    """Test RelativePositionEmbed forward pass output shape.

    Args:
        device: Device for computation.
    """
    window_size = (8, 8)
    batch_size = 2
    num_heads = 4
    num_windows_h = 7
    num_windows_w = 10

    rel_pos_embed = RelativePositionEmbed(window_size).to(device)

    wh, ww = window_size
    # Input shape: (B, H, Hw, Ww, wh*ww, wh*ww)
    x = torch.randn(
        batch_size,
        num_heads,
        num_windows_h,
        num_windows_w,
        wh * ww,
        wh * ww,
    ).to(device)

    output = rel_pos_embed(x)

    assert output.shape == x.shape
    assert not torch.isnan(output).any()


def test_relative_position_embed_forward_adds_embedding(
    device: str,
) -> None:
    """Test RelativePositionEmbed forward pass adds embedding.

    Args:
        device: Device for computation.
    """
    window_size = (4, 4)
    batch_size = 2
    num_heads = 2
    num_windows_h = 4
    num_windows_w = 4

    rel_pos_embed = RelativePositionEmbed(window_size).to(device)

    wh, ww = window_size
    x = torch.randn(
        batch_size,
        num_heads,
        num_windows_h,
        num_windows_w,
        wh * ww,
        wh * ww,
    ).to(device)

    output = rel_pos_embed(x)

    # Output should be different from input (embedding added)
    assert not torch.allclose(output, x)


def test_relative_position_embed_rectangular_window() -> None:
    """Test RelativePositionEmbed with rectangular window."""
    window_size = (8, 4)
    rel_pos_embed = RelativePositionEmbed(window_size)

    wh, ww = window_size
    expected_shape = (2 * wh - 1, 2 * ww - 1)
    assert rel_pos_embed.pos_embeddings.shape == expected_shape

    # Check indices
    assert rel_pos_embed.relative_indices.shape == (wh * ww, wh * ww, 2)


# ============================================================================
# Tests for TimeEmbed
# ============================================================================


def test_time_embed_init() -> None:
    """Test TimeEmbed initialization."""
    embed_dim = 64
    time_embed = TimeEmbed(embed_dim)

    assert isinstance(time_embed, TimeEmbed)
    assert isinstance(time_embed.lead_time_embed, nn.Linear)
    assert time_embed.lead_time_embed.in_features == 1
    assert time_embed.lead_time_embed.out_features == embed_dim


def test_time_embed_forward_shape(device: str) -> None:
    """Test TimeEmbed forward pass output shape.

    Args:
        device: Device for computation.
    """
    embed_dim = 64
    batch_size = 3
    num_tokens = 64

    time_embed = TimeEmbed(embed_dim).to(device)

    # Input shape: (B, L, D)
    x = torch.randn(batch_size, num_tokens, embed_dim).to(device)
    lead_times = torch.rand(batch_size).to(device)

    output = time_embed(x, lead_times)

    assert output.shape == x.shape
    assert not torch.isnan(output).any()


def test_time_embed_forward_adds_embedding(device: str) -> None:
    """Test TimeEmbed forward pass adds embedding to input.

    Args:
        device: Device for computation.
    """
    embed_dim = 64
    batch_size = 2
    num_tokens = 32

    time_embed = TimeEmbed(embed_dim).to(device)

    x = torch.randn(batch_size, num_tokens, embed_dim).to(device)
    lead_times = torch.rand(batch_size).to(device)

    output = time_embed(x, lead_times)

    # Output should be different from input (embedding added)
    assert not torch.allclose(output, x)


def test_time_embed_different_lead_times(device: str) -> None:
    """Test TimeEmbed with different lead time values.

    Args:
        device: Device for computation.
    """
    embed_dim = 64
    batch_size = 3
    num_tokens = 32

    time_embed = TimeEmbed(embed_dim).to(device)

    x = torch.randn(batch_size, num_tokens, embed_dim).to(device)

    # Test with different lead time ranges
    lead_times_1 = torch.tensor([0.0, 0.5, 1.0]).to(device)
    lead_times_2 = torch.tensor([1.0, 2.0, 3.0]).to(device)

    output_1 = time_embed(x.clone(), lead_times_1)
    output_2 = time_embed(x.clone(), lead_times_2)

    # Different lead times should produce different outputs
    assert not torch.allclose(output_1, output_2)


# ============================================================================
# Tests for DiffusionTimeEmbed
# ============================================================================


def test_diffusion_time_embed_init() -> None:
    """Test DiffusionTimeEmbed initialization."""
    embed_dim = 64
    diff_time_embed = DiffusionTimeEmbed(embed_dim)

    assert isinstance(diff_time_embed, DiffusionTimeEmbed)
    assert isinstance(diff_time_embed.diff_time_embed, nn.Linear)
    assert diff_time_embed.diff_time_embed.in_features == 1
    assert diff_time_embed.diff_time_embed.out_features == embed_dim


def test_diffusion_time_embed_forward_shape(device: str) -> None:
    """Test DiffusionTimeEmbed forward pass output shape.

    Args:
        device: Device for computation.
    """
    embed_dim = 64
    batch_size = 3
    num_tokens = 64

    diff_time_embed = DiffusionTimeEmbed(embed_dim).to(device)

    # Input shape: (B, L, D)
    x = torch.randn(batch_size, num_tokens, embed_dim).to(device)
    diff_times = torch.rand(batch_size).to(device)

    output = diff_time_embed(x, diff_times)

    assert output.shape == x.shape
    assert not torch.isnan(output).any()


def test_diffusion_time_embed_forward_adds_embedding(device: str) -> None:
    """Test DiffusionTimeEmbed forward pass adds embedding to input.

    Args:
        device: Device for computation.
    """
    embed_dim = 64
    batch_size = 2
    num_tokens = 32

    diff_time_embed = DiffusionTimeEmbed(embed_dim).to(device)

    x = torch.randn(batch_size, num_tokens, embed_dim).to(device)
    diff_times = torch.rand(batch_size).to(device)

    output = diff_time_embed(x, diff_times)

    # Output should be different from input (embedding added)
    assert not torch.allclose(output, x)


def test_diffusion_time_embed_tau_range(device: str) -> None:
    """Test DiffusionTimeEmbed with tau in [0, 1] range.

    Args:
        device: Device for computation.
    """
    embed_dim = 64
    batch_size = 3
    num_tokens = 32

    diff_time_embed = DiffusionTimeEmbed(embed_dim).to(device)

    x = torch.randn(batch_size, num_tokens, embed_dim).to(device)

    # Test with tau values in [0, 1]
    diff_times = torch.tensor([0.0, 0.5, 1.0]).to(device)

    output = diff_time_embed(x, diff_times)

    assert output.shape == x.shape
    assert not torch.isnan(output).any()


def test_diffusion_time_embed_different_tau_values(device: str) -> None:
    """Test DiffusionTimeEmbed with different tau values.

    Args:
        device: Device for computation.
    """
    embed_dim = 64
    batch_size = 3
    num_tokens = 32

    diff_time_embed = DiffusionTimeEmbed(embed_dim).to(device)

    x = torch.randn(batch_size, num_tokens, embed_dim).to(device)

    # Test with different tau values
    tau_1 = torch.tensor([0.0, 0.3, 0.6]).to(device)
    tau_2 = torch.tensor([0.2, 0.5, 0.8]).to(device)

    output_1 = diff_time_embed(x.clone(), tau_1)
    output_2 = diff_time_embed(x.clone(), tau_2)

    # Different tau values should produce different outputs
    assert not torch.allclose(output_1, output_2)


# ============================================================================
# Edge case and integration tests
# ============================================================================


def test_var_embed_single_variable(default_vars: list[str], device: str) -> None:
    """Test VarEmbed with single variable.

    Args:
        default_vars: List of variable names.
        device: Device for computation.
    """
    embed_dim = 64
    batch_size = 2
    num_tokens = 32

    var_embed = VarEmbed(default_vars, embed_dim).to(device)

    # Single variable
    x = torch.randn(batch_size, 1, num_tokens, embed_dim).to(device)
    in_vars = torch.tensor([0]).to(device)

    output = var_embed(x, in_vars)

    assert output.shape == x.shape
    assert not torch.isnan(output).any()


def test_pos_embed_single_patch(device: str) -> None:
    """Test PosEmbed with single patch.

    Args:
        device: Device for computation.
    """
    embed_dim = 64
    patch_size = (128, 128)
    image_size = (128, 128)
    num_patches = 1
    batch_size = 2

    pos_embed = PosEmbed(embed_dim, patch_size, image_size, num_patches).to(device)

    x = torch.randn(batch_size, num_patches, embed_dim).to(device)

    output = pos_embed(x)

    assert output.shape == x.shape
    assert not torch.isnan(output).any()


def test_time_embed_zero_lead_time(device: str) -> None:
    """Test TimeEmbed with zero lead time.

    Args:
        device: Device for computation.
    """
    embed_dim = 64
    batch_size = 2
    num_tokens = 32

    time_embed = TimeEmbed(embed_dim).to(device)

    x = torch.randn(batch_size, num_tokens, embed_dim).to(device)
    lead_times = torch.zeros(batch_size).to(device)

    output = time_embed(x, lead_times)

    assert output.shape == x.shape
    assert not torch.isnan(output).any()


def test_diffusion_time_embed_boundary_values(device: str) -> None:
    """Test DiffusionTimeEmbed at tau boundary values (0 and 1).

    Args:
        device: Device for computation.
    """
    embed_dim = 64
    batch_size = 2
    num_tokens = 32

    diff_time_embed = DiffusionTimeEmbed(embed_dim).to(device)

    x = torch.randn(batch_size, num_tokens, embed_dim).to(device)

    # Test tau = 0
    tau_zero = torch.zeros(batch_size).to(device)
    output_zero = diff_time_embed(x.clone(), tau_zero)
    assert not torch.isnan(output_zero).any()

    # Test tau = 1
    tau_one = torch.ones(batch_size).to(device)
    output_one = diff_time_embed(x.clone(), tau_one)
    assert not torch.isnan(output_one).any()

    # Outputs should be different
    assert not torch.allclose(output_zero, output_one)


def test_embedding_combination(default_vars: list[str], device: str) -> None:
    """Test combining multiple embeddings sequentially.

    Args:
        default_vars: List of variable names.
        device: Device for computation.
    """
    embed_dim = 64
    patch_size = (16, 16)
    image_size = (128, 128)
    num_patches = 64
    batch_size = 2
    num_vars = 4
    num_tokens = 64

    # Create all embedding modules
    var_embed = VarEmbed(default_vars, embed_dim).to(device)
    pos_embed = PosEmbed(embed_dim, patch_size, image_size, num_patches).to(device)
    time_embed = TimeEmbed(embed_dim).to(device)
    diff_time_embed = DiffusionTimeEmbed(embed_dim).to(device)

    # Start with variable embedding
    x = torch.randn(batch_size, num_vars, num_tokens, embed_dim).to(device)
    in_vars = torch.tensor([0, 1, 2, 3]).to(device)
    x = var_embed(x, in_vars)

    # Simulate variable aggregation (reduce variable dimension)
    x = x.mean(dim=1)  # (B, L, D)

    # Add position embedding
    x = pos_embed(x)

    # Add temporal embedding
    lead_times = torch.rand(batch_size).to(device)
    x = time_embed(x, lead_times)

    # Add diffusion time embedding
    diff_times = torch.rand(batch_size).to(device)
    x = diff_time_embed(x, diff_times)

    assert x.shape == (batch_size, num_tokens, embed_dim)
    assert not torch.isnan(x).any()
