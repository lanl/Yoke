"""Tests for the LodeRunnerViT architecture.

This module contains unit tests for :class:`LodeRunnerViT`, the LodeRunner
variant that swaps the SWIN-V2 U-Net backbone for an isotropic plain-ViT
backbone with 2-D RoPE attention.

Tests cover initialization, the embed-dim/head constraint, forward-pass shape
behavior, output-variable selection, noise injection, gradient flow, and both
branch-fusion modes.
"""

import pytest
import torch

from yoke.models.vit.plain_vit import PlainViTBackbone
from yoke.models.vit.swin.bomberman import LodeRunnerViT


@pytest.fixture
def device() -> str:
    """Get device for testing.

    Returns:
        Device string ('cuda' or 'cpu').
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def default_vars() -> list[str]:
    """Create default variable list fixture.

    Returns:
        List of variable names for testing.
    """
    return ["var1", "var2", "var3", "var4", "var5"]


@pytest.fixture
def loderunner_vit(default_vars: list[str], device: str) -> LodeRunnerViT:
    """Fixture for a small LodeRunnerViT model.

    Args:
        default_vars: List of variable names.
        device: Device for computation.

    Returns:
        An initialized LodeRunnerViT model.
    """
    return LodeRunnerViT(
        default_vars=default_vars,
        image_size=(1120, 800),
        patch_size=(10, 10),
        embed_dim=128,
        num_heads=8,
        num_attention_heads=8,
        attention_head_dim=16,
        num_layers=2,
        mlp_ratio=4.0,
        concat_mlp=True,
    ).to(device)


# ============================================================================
# Initialization tests
# ============================================================================


def test_loderunner_vit_init(loderunner_vit: LodeRunnerViT) -> None:
    """Test LodeRunnerViT initialization and derived attributes.

    Args:
        loderunner_vit: The model fixture.
    """
    assert isinstance(loderunner_vit, LodeRunnerViT)
    assert loderunner_vit.embed_dim == 128
    assert loderunner_vit.max_vars == 5
    assert loderunner_vit.num_layers == 2
    assert isinstance(loderunner_vit.backbone, PlainViTBackbone)


def test_loderunner_vit_no_pos_embed(loderunner_vit: LodeRunnerViT) -> None:
    """Test LodeRunnerViT omits the additive position embedding.

    Args:
        loderunner_vit: The model fixture.
    """
    # Position is injected via RoPE inside attention, not an additive PosEmbed.
    assert not hasattr(loderunner_vit, "pos_embed")


def test_loderunner_vit_backbone_grid_matches_embed(
    loderunner_vit: LodeRunnerViT,
) -> None:
    """Test the backbone patch grid matches the parallel embedding grid.

    Args:
        loderunner_vit: The model fixture.
    """
    assert (
        loderunner_vit.backbone.patch_grid_size
        == loderunner_vit.parallel_embed.grid_size
    )


def test_loderunner_vit_bad_embed_dim_raises(
    default_vars: list[str], device: str
) -> None:
    """Test LodeRunnerViT rejects embed_dim != heads * head_dim.

    Args:
        default_vars: List of variable names.
        device: Device for computation.
    """
    with pytest.raises(AssertionError, match="embed_dim must equal"):
        LodeRunnerViT(
            default_vars=default_vars,
            image_size=(1120, 800),
            patch_size=(10, 10),
            embed_dim=100,  # != 8 * 16
            num_attention_heads=8,
            attention_head_dim=16,
            num_layers=1,
        ).to(device)


# ============================================================================
# Forward-pass tests
# ============================================================================


def test_loderunner_vit_forward_shape(
    loderunner_vit: LodeRunnerViT, device: str
) -> None:
    """Test LodeRunnerViT forward output shape.

    Args:
        loderunner_vit: The model fixture.
        device: Device for computation.
    """
    x = torch.randn(2, 3, 1120, 800).to(device)
    in_vars = torch.tensor([0, 1, 2]).to(device)
    out_vars = torch.tensor([0, 1, 2]).to(device)
    lead_times = torch.rand(2).to(device)

    output = loderunner_vit(x, in_vars, out_vars, lead_times)

    assert output.shape == (2, 3, 1120, 800)
    assert not torch.isnan(output).any()


def test_loderunner_vit_output_var_selection(
    loderunner_vit: LodeRunnerViT, device: str
) -> None:
    """Test LodeRunnerViT selects only requested output variables.

    Args:
        loderunner_vit: The model fixture.
        device: Device for computation.
    """
    x = torch.randn(2, 3, 1120, 800).to(device)
    in_vars = torch.tensor([0, 1, 2]).to(device)
    out_vars = torch.tensor([0, 1]).to(device)  # fewer outputs than inputs
    lead_times = torch.rand(2).to(device)

    output = loderunner_vit(x, in_vars, out_vars, lead_times)

    assert output.shape[0] == 2
    assert output.shape[1] == len(out_vars)


def test_loderunner_vit_forward_identical_signature(
    loderunner_vit: LodeRunnerViT, device: str
) -> None:
    """Test LodeRunnerViT accepts the LodeRunner forward signature.

    Args:
        loderunner_vit: The model fixture.
        device: Device for computation.
    """
    x = torch.randn(1, 2, 1120, 800).to(device)
    in_vars = torch.tensor([2, 4]).to(device)
    out_vars = torch.tensor([2, 4]).to(device)
    lead_times = torch.rand(1).to(device)

    output = loderunner_vit(x, in_vars, out_vars, lead_times)

    assert output.shape == (1, 2, 1120, 800)


def test_loderunner_vit_noise_injection(
    default_vars: list[str], device: str
) -> None:
    """Test nonzero noise_scale changes the output stochastically.

    Args:
        default_vars: List of variable names.
        device: Device for computation.
    """
    model = LodeRunnerViT(
        default_vars=default_vars,
        image_size=(1120, 800),
        patch_size=(10, 10),
        embed_dim=128,
        num_attention_heads=8,
        attention_head_dim=16,
        num_layers=1,
        noise_scale=1.0,
    ).to(device)
    model.eval()

    x = torch.randn(2, 3, 1120, 800).to(device)
    in_vars = torch.tensor([0, 1, 2]).to(device)
    out_vars = torch.tensor([0, 1, 2]).to(device)
    lead_times = torch.rand(2).to(device)

    torch.manual_seed(0)
    out1 = model(x, in_vars, out_vars, lead_times)
    torch.manual_seed(1)
    out2 = model(x, in_vars, out_vars, lead_times)

    # Different noise draws produce different outputs.
    assert not torch.allclose(out1, out2)


@pytest.mark.parametrize("concat_mlp", [True, False])
def test_loderunner_vit_fusion_modes(
    default_vars: list[str], device: str, concat_mlp: bool
) -> None:
    """Test LodeRunnerViT runs with both branch-fusion modes.

    Args:
        default_vars: List of variable names.
        device: Device for computation.
        concat_mlp: Whether to use concat-then-project fusion.
    """
    model = LodeRunnerViT(
        default_vars=default_vars,
        image_size=(1120, 800),
        patch_size=(10, 10),
        embed_dim=128,
        num_attention_heads=8,
        attention_head_dim=16,
        num_layers=1,
        concat_mlp=concat_mlp,
    ).to(device)

    x = torch.randn(1, 2, 1120, 800).to(device)
    in_vars = torch.tensor([0, 1]).to(device)
    out_vars = torch.tensor([0, 1]).to(device)
    lead_times = torch.rand(1).to(device)

    output = model(x, in_vars, out_vars, lead_times)

    assert output.shape == (1, 2, 1120, 800)


def test_loderunner_vit_verbose(default_vars: list[str], device: str) -> None:
    """Test LodeRunnerViT verbose construction succeeds.

    Args:
        default_vars: List of variable names.
        device: Device for computation.
    """
    model = LodeRunnerViT(
        default_vars=default_vars,
        image_size=(1120, 800),
        patch_size=(10, 10),
        embed_dim=128,
        num_attention_heads=8,
        attention_head_dim=16,
        num_layers=1,
        verbose=True,
    ).to(device)

    assert isinstance(model, LodeRunnerViT)


def test_loderunner_vit_gradient_flow(
    loderunner_vit: LodeRunnerViT, device: str
) -> None:
    """Test gradients flow through all LodeRunnerViT parameters.

    Args:
        loderunner_vit: The model fixture.
        device: Device for computation.
    """
    x = torch.randn(2, 3, 1120, 800).to(device)
    in_vars = torch.tensor([0, 1, 2]).to(device)
    out_vars = torch.tensor([0, 1, 2]).to(device)
    lead_times = torch.rand(2).to(device)

    loss = loderunner_vit(x, in_vars, out_vars, lead_times).pow(2).mean()
    loss.backward()

    for name, param in loderunner_vit.named_parameters():
        # The parallel embedding stores one weight bank per default variable;
        # only banks for the active in_vars receive gradients.
        if "parallel_embed" in name:
            continue
        assert param.grad is not None, f"No gradient for {name}"


# ============================================================================
# eps / bias forwarding (plan Section 1.4)
# ============================================================================


def test_loderunner_vit_eps_forwarded(default_vars: list[str], device: str) -> None:
    """Test the eps argument is threaded into the backbone LayerNorm/RMSNorm."""
    model = LodeRunnerViT(
        default_vars=default_vars,
        image_size=(1120, 800),
        patch_size=(10, 10),
        embed_dim=128,
        num_attention_heads=8,
        attention_head_dim=16,
        num_layers=2,
        eps=1e-7,
    ).to(device)

    assert model.eps == 1e-7
    assert model.backbone.norm_out.eps == 1e-7
    assert model.backbone.transformer_blocks[0].norm.eps == 1e-7
    assert model.backbone.transformer_blocks[0].attn.norm_q.eps == 1e-7


def test_loderunner_vit_bias_default_off(default_vars: list[str], device: str) -> None:
    """Test bias defaults to False (bias-free) for backward compatibility."""
    model = LodeRunnerViT(
        default_vars=default_vars,
        image_size=(1120, 800),
        patch_size=(10, 10),
        embed_dim=128,
        num_attention_heads=8,
        attention_head_dim=16,
        num_layers=1,
    ).to(device)

    attn = model.backbone.transformer_blocks[0].attn
    assert attn.to_q.bias is None
    assert attn.to_out[0].bias is None


def test_loderunner_vit_bias_on(default_vars: list[str], device: str) -> None:
    """Test bias=True enables Q/K/V and output-projection biases."""
    model = LodeRunnerViT(
        default_vars=default_vars,
        image_size=(1120, 800),
        patch_size=(10, 10),
        embed_dim=128,
        num_attention_heads=8,
        attention_head_dim=16,
        num_layers=1,
        bias=True,
    ).to(device)

    attn = model.backbone.transformer_blocks[0].attn
    assert attn.to_q.bias is not None
    assert attn.to_k.bias is not None
    assert attn.to_v.bias is not None
    assert attn.to_out[0].bias is not None


# ============================================================================
# 2-in / 1-out extension (plan Section 3)
# ============================================================================


def test_loderunner_vit_rejects_bad_num_input_frames(
    default_vars: list[str], device: str
) -> None:
    """Test num_input_frames must be 1 or 2."""
    with pytest.raises(AssertionError, match="num_input_frames"):
        LodeRunnerViT(
            default_vars=default_vars,
            image_size=(1120, 800),
            patch_size=(10, 10),
            embed_dim=128,
            num_attention_heads=8,
            attention_head_dim=16,
            num_layers=1,
            num_input_frames=3,
        ).to(device)


def test_loderunner_vit_single_frame_has_no_fusion(
    loderunner_vit: LodeRunnerViT,
) -> None:
    """Test the single-frame model does not build the temporal fusion layer."""
    assert loderunner_vit.num_input_frames == 1
    assert loderunner_vit.temporal_fusion is None


@pytest.fixture
def loderunner_vit_2frame(default_vars: list[str], device: str) -> LodeRunnerViT:
    """Fixture for a small 2-in/1-out LodeRunnerViT model."""
    return LodeRunnerViT(
        default_vars=default_vars,
        image_size=(1120, 800),
        patch_size=(10, 10),
        embed_dim=128,
        num_heads=8,
        num_attention_heads=8,
        attention_head_dim=16,
        num_layers=2,
        num_input_frames=2,
    ).to(device)


def test_loderunner_vit_2frame_builds_fusion(
    loderunner_vit_2frame: LodeRunnerViT,
) -> None:
    """Test the 2-frame model builds a 2D->D temporal fusion linear."""
    assert loderunner_vit_2frame.num_input_frames == 2
    fusion = loderunner_vit_2frame.temporal_fusion
    assert isinstance(fusion, torch.nn.Linear)
    assert fusion.in_features == 2 * loderunner_vit_2frame.embed_dim
    assert fusion.out_features == loderunner_vit_2frame.embed_dim


def test_loderunner_vit_2frame_forward_shape(
    loderunner_vit_2frame: LodeRunnerViT, device: str
) -> None:
    """Test the 2-frame forward produces a single-frame prediction."""
    x = torch.randn(2, 2, 3, 1120, 800).to(device)  # (B, T=2, C, H, W)
    in_vars = torch.tensor([0, 1, 2]).to(device)
    out_vars = torch.tensor([0, 1, 2]).to(device)
    lead_times = torch.rand(2, 2).to(device)  # (dt_in, dt_out)

    output = loderunner_vit_2frame(x, in_vars, out_vars, lead_times)

    assert output.shape == (2, 3, 1120, 800)
    assert not torch.isnan(output).any()


def test_loderunner_vit_2frame_bad_input_shape_raises(
    loderunner_vit_2frame: LodeRunnerViT, device: str
) -> None:
    """Test the 2-frame path rejects a single-frame input tensor."""
    x = torch.randn(2, 3, 1120, 800).to(device)  # missing time axis
    in_vars = torch.tensor([0, 1, 2]).to(device)
    out_vars = torch.tensor([0, 1, 2]).to(device)
    lead_times = torch.rand(2, 2).to(device)

    with pytest.raises(AssertionError, match="B, 2, C, H, W"):
        loderunner_vit_2frame(x, in_vars, out_vars, lead_times)


def test_loderunner_vit_2frame_bad_leadtime_shape_raises(
    loderunner_vit_2frame: LodeRunnerViT, device: str
) -> None:
    """Test the 2-frame path rejects scalar lead-times."""
    x = torch.randn(2, 2, 3, 1120, 800).to(device)
    in_vars = torch.tensor([0, 1, 2]).to(device)
    out_vars = torch.tensor([0, 1, 2]).to(device)
    lead_times = torch.rand(2).to(device)  # should be (B, 2)

    with pytest.raises(AssertionError, match="dt_in, dt_out"):
        loderunner_vit_2frame(x, in_vars, out_vars, lead_times)


def test_loderunner_vit_2frame_gradient_flow(
    loderunner_vit_2frame: LodeRunnerViT, device: str
) -> None:
    """Test gradients flow through the 2-frame model, incl. temporal fusion."""
    x = torch.randn(2, 2, 3, 1120, 800).to(device)
    in_vars = torch.tensor([0, 1, 2]).to(device)
    out_vars = torch.tensor([0, 1, 2]).to(device)
    lead_times = torch.rand(2, 2).to(device)

    loss = loderunner_vit_2frame(x, in_vars, out_vars, lead_times).pow(2).mean()
    loss.backward()

    assert loderunner_vit_2frame.temporal_fusion.weight.grad is not None


def test_loderunner_vit_single_frame_regression(
    default_vars: list[str], device: str
) -> None:
    """Test that num_input_frames=1 (default) reproduces the prior forward path.

    With noise disabled and a fixed seed, the single-frame path must be
    deterministic and unchanged in interface: (B, C, H, W) input, (B,)
    lead-times.
    """
    model = LodeRunnerViT(
        default_vars=default_vars,
        image_size=(1120, 800),
        patch_size=(10, 10),
        embed_dim=128,
        num_attention_heads=8,
        attention_head_dim=16,
        num_layers=2,
        noise_scale=0.0,
    ).to(device)
    model.eval()

    x = torch.randn(2, 3, 1120, 800).to(device)
    in_vars = torch.tensor([0, 1, 2]).to(device)
    out_vars = torch.tensor([0, 1, 2]).to(device)
    lead_times = torch.rand(2).to(device)

    out1 = model(x, in_vars, out_vars, lead_times)
    out2 = model(x, in_vars, out_vars, lead_times)

    # Deterministic with noise off.
    assert torch.allclose(out1, out2)
    assert out1.shape == (2, 3, 1120, 800)

