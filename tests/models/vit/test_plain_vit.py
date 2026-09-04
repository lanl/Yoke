"""Tests for the plain Vision-Transformer backbone.

This module contains unit tests for the plain-ViT transformer components used
as a swappable backbone in LodeRunner:

- RopeAttention: global self-attention with per-head QK RMSNorm and 2-D RoPE.
- BasicTransformerBlock: parallel attention + SwiGLU-MLP block.
- PlainViTBackbone: isotropic (B, N, D) -> (B, N, D) transformer stack.

Tests cover initialization, forward passes, shape validation, fusion modes,
gradient flow, and assertion/error handling.
"""

import pytest
import torch
import torch.nn as nn

from yoke.models.vit.plain_vit import (
    BasicTransformerBlock,
    PlainViTBackbone,
    RopeAttention,
)
from yoke.models.vit.rope import RotaryPositionalEmbeddingFromCenters


@pytest.fixture
def device() -> str:
    """Get device for testing.

    Returns:
        Device string ('cuda' or 'cpu').
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def _make_rotary(
    num_tokens: int, head_dim: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a (cos, sin) rotary embedding for a token sequence.

    Args:
        num_tokens: Number of tokens N.
        head_dim: Attention head dimension.
        device: Device for computation.

    Returns:
        Tuple of (cos, sin) tensors each of shape (N, head_dim).
    """
    rope = RotaryPositionalEmbeddingFromCenters(
        rope_dim_list=[head_dim // 2, head_dim // 2]
    ).to(device)
    centers = torch.rand(num_tokens, 2, device=device)
    return rope(centers)


# ============================================================================
# Tests for RopeAttention
# ============================================================================


def test_rope_attention_init() -> None:
    """Test RopeAttention initialization."""
    attn = RopeAttention(query_dim=128, heads=8, dim_head=16)

    assert attn.heads == 8
    assert attn.dim_head == 16
    assert attn.inner_dim == 128
    assert isinstance(attn.to_q, nn.Linear)
    assert isinstance(attn.norm_q, nn.RMSNorm)
    assert isinstance(attn.norm_k, nn.RMSNorm)


def test_rope_attention_bias_flags() -> None:
    """Test RopeAttention respects projection bias flags."""
    attn = RopeAttention(query_dim=64, heads=4, dim_head=16, bias=False, out_bias=True)

    assert attn.to_q.bias is None
    assert attn.to_out[0].bias is not None


def test_rope_attention_forward_shape(device: str) -> None:
    """Test RopeAttention forward pass output shape.

    Args:
        device: Device for computation.
    """
    query_dim, heads, dim_head = 128, 8, 16
    N = 20
    attn = RopeAttention(query_dim=query_dim, heads=heads, dim_head=dim_head).to(device)
    x = torch.randn(2, N, query_dim, device=device)
    rotary = _make_rotary(N, dim_head, device)

    out = attn(x, image_rotary_emb=rotary)

    assert out.shape == (2, N, query_dim)
    assert not torch.isnan(out).any()


def test_rope_attention_forward_no_rotary(device: str) -> None:
    """Test RopeAttention forward pass without rotary embedding.

    Args:
        device: Device for computation.
    """
    query_dim, heads, dim_head = 64, 4, 16
    N = 12
    attn = RopeAttention(query_dim=query_dim, heads=heads, dim_head=dim_head).to(device)
    x = torch.randn(2, N, query_dim, device=device)

    out = attn(x, image_rotary_emb=None)

    assert out.shape == (2, N, query_dim)
    assert not torch.isnan(out).any()


def test_rope_attention_dropout_eval_vs_train(device: str) -> None:
    """Test RopeAttention dropout is inactive in eval mode.

    Args:
        device: Device for computation.
    """
    query_dim, heads, dim_head = 64, 4, 16
    N = 10
    attn = RopeAttention(
        query_dim=query_dim, heads=heads, dim_head=dim_head, dropout=0.5
    ).to(device)
    attn.eval()
    x = torch.randn(2, N, query_dim, device=device)

    out1 = attn(x)
    out2 = attn(x)

    # Deterministic in eval mode (no dropout).
    assert torch.allclose(out1, out2)


# ============================================================================
# Tests for BasicTransformerBlock
# ============================================================================


def test_basic_block_init_concat() -> None:
    """Test BasicTransformerBlock concat-fusion initialization."""
    block = BasicTransformerBlock(
        num_attention_heads=8, attention_head_dim=16, mlp_ratio=4.0, concat=True
    )

    hidden = 8 * 16
    mlp_dim = int(hidden * 4.0)
    assert block.concat is True
    assert block.proj_out.in_features == hidden + mlp_dim
    assert block.proj_out.out_features == hidden
    assert not hasattr(block, "proj_mlp_down")


def test_basic_block_init_sum() -> None:
    """Test BasicTransformerBlock sum-fusion initialization."""
    block = BasicTransformerBlock(
        num_attention_heads=8, attention_head_dim=16, concat=False
    )

    hidden = 8 * 16
    assert block.concat is False
    assert isinstance(block.proj_mlp_down, nn.Linear)
    assert block.proj_out.in_features == hidden
    assert block.proj_out.out_features == hidden


def test_basic_block_non_affine_norm() -> None:
    """Test BasicTransformerBlock uses a non-affine LayerNorm."""
    block = BasicTransformerBlock(num_attention_heads=4, attention_head_dim=16)

    assert isinstance(block.norm, nn.LayerNorm)
    assert block.norm.elementwise_affine is False


@pytest.mark.parametrize("concat", [True, False])
def test_basic_block_forward_shape(device: str, concat: bool) -> None:
    """Test BasicTransformerBlock forward pass output shape.

    Args:
        device: Device for computation.
        concat: Whether to use concat-then-project fusion.
    """
    heads, dim_head = 8, 16
    hidden = heads * dim_head
    N = 15
    block = BasicTransformerBlock(
        num_attention_heads=heads, attention_head_dim=dim_head, concat=concat
    ).to(device)
    x = torch.randn(2, N, hidden, device=device)
    rotary = _make_rotary(N, dim_head, device)

    out = block(x, image_rotary_emb=rotary)

    assert out.shape == (2, N, hidden)
    assert not torch.isnan(out).any()


def test_basic_block_residual(device: str) -> None:
    """Test BasicTransformerBlock applies a residual update.

    Args:
        device: Device for computation.
    """
    heads, dim_head = 4, 16
    hidden = heads * dim_head
    N = 10
    block = BasicTransformerBlock(
        num_attention_heads=heads, attention_head_dim=dim_head
    ).to(device)

    # Zero the fusion projection so the block output equals the residual input.
    with torch.no_grad():
        block.proj_out.weight.zero_()
        block.proj_out.bias.zero_()

    x = torch.randn(2, N, hidden, device=device)
    rotary = _make_rotary(N, dim_head, device)
    out = block(x, image_rotary_emb=rotary)

    assert torch.allclose(out, x, atol=1e-6)


# ============================================================================
# Tests for PlainViTBackbone
# ============================================================================


def test_plain_vit_backbone_init() -> None:
    """Test PlainViTBackbone initialization and derived dimensions."""
    backbone = PlainViTBackbone(
        patch_grid_size=(112, 80),
        num_attention_heads=8,
        attention_head_dim=16,
        num_layers=6,
    )

    assert backbone.emb_size == 128
    assert backbone.num_patches == 112 * 80
    assert len(backbone.transformer_blocks) == 6
    assert backbone.centers.shape == (112 * 80, 2)
    assert isinstance(backbone.norm_out, nn.LayerNorm)


def test_plain_vit_backbone_odd_head_dim_raises() -> None:
    """Test PlainViTBackbone requires an even attention_head_dim."""
    with pytest.raises(AssertionError, match="attention_head_dim must be even"):
        PlainViTBackbone(
            patch_grid_size=(8, 8),
            num_attention_heads=4,
            attention_head_dim=15,
            num_layers=2,
        )


def test_plain_vit_backbone_forward_shape(device: str) -> None:
    """Test PlainViTBackbone forward preserves (B, N, D) shape.

    Args:
        device: Device for computation.
    """
    grid = (16, 12)
    heads, dim_head = 8, 16
    emb = heads * dim_head
    backbone = PlainViTBackbone(
        patch_grid_size=grid,
        num_attention_heads=heads,
        attention_head_dim=dim_head,
        num_layers=3,
    ).to(device)

    x = torch.randn(2, grid[0] * grid[1], emb, device=device)
    out = backbone(x)

    assert out.shape == x.shape
    assert not torch.isnan(out).any()


def test_plain_vit_backbone_verbose(device: str, capsys: pytest.CaptureFixture) -> None:
    """Test PlainViTBackbone verbose construction prints dimensions.

    Args:
        device: Device for computation.
        capsys: Pytest capture fixture.
    """
    PlainViTBackbone(
        patch_grid_size=(8, 8),
        num_attention_heads=4,
        attention_head_dim=16,
        num_layers=2,
        verbose=True,
    ).to(device)

    captured = capsys.readouterr()
    assert "PlainViTBackbone patch-grid size" in captured.out
    assert "rope_dim_list" in captured.out


@pytest.mark.parametrize("concat_mlp", [True, False])
def test_plain_vit_backbone_fusion_modes(device: str, concat_mlp: bool) -> None:
    """Test PlainViTBackbone works with both fusion modes.

    Args:
        device: Device for computation.
        concat_mlp: Whether to use concat-then-project fusion.
    """
    grid = (10, 10)
    heads, dim_head = 4, 16
    emb = heads * dim_head
    backbone = PlainViTBackbone(
        patch_grid_size=grid,
        num_attention_heads=heads,
        attention_head_dim=dim_head,
        num_layers=2,
        concat_mlp=concat_mlp,
    ).to(device)

    x = torch.randn(2, grid[0] * grid[1], emb, device=device)
    out = backbone(x)

    assert out.shape == x.shape


def test_plain_vit_backbone_bad_token_count_raises(device: str) -> None:
    """Test PlainViTBackbone rejects a mismatched token count.

    Args:
        device: Device for computation.
    """
    grid = (10, 10)
    emb = 4 * 16
    backbone = PlainViTBackbone(
        patch_grid_size=grid,
        num_attention_heads=4,
        attention_head_dim=16,
        num_layers=1,
    ).to(device)

    x = torch.randn(2, 50, emb, device=device)  # wrong N (expected 100)
    with pytest.raises(AssertionError, match="does not match patch grid"):
        backbone(x)


def test_plain_vit_backbone_bad_emb_dim_raises(device: str) -> None:
    """Test PlainViTBackbone rejects a mismatched embedding dim.

    Args:
        device: Device for computation.
    """
    grid = (10, 10)
    backbone = PlainViTBackbone(
        patch_grid_size=grid,
        num_attention_heads=4,
        attention_head_dim=16,
        num_layers=1,
    ).to(device)

    x = torch.randn(2, 100, 99, device=device)  # wrong D (expected 64)
    with pytest.raises(AssertionError, match="does not match backbone embedding dim"):
        backbone(x)


def test_plain_vit_backbone_gradient_flow(device: str) -> None:
    """Test gradients flow through all PlainViTBackbone parameters.

    Args:
        device: Device for computation.
    """
    grid = (8, 8)
    heads, dim_head = 4, 16
    emb = heads * dim_head
    backbone = PlainViTBackbone(
        patch_grid_size=grid,
        num_attention_heads=heads,
        attention_head_dim=dim_head,
        num_layers=2,
    ).to(device)

    x = torch.randn(2, grid[0] * grid[1], emb, device=device)
    loss = backbone(x).pow(2).mean()
    loss.backward()

    for name, param in backbone.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


def test_plain_vit_backbone_depth_scaling(device: str) -> None:
    """Test increasing num_layers increases parameter count.

    Args:
        device: Device for computation.
    """
    kwargs = dict(
        patch_grid_size=(8, 8),
        num_attention_heads=4,
        attention_head_dim=16,
    )
    shallow = PlainViTBackbone(num_layers=2, **kwargs)
    deep = PlainViTBackbone(num_layers=6, **kwargs)

    n_shallow = sum(p.numel() for p in shallow.parameters())
    n_deep = sum(p.numel() for p in deep.parameters())
    assert n_deep > n_shallow
