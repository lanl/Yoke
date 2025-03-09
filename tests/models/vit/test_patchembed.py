"""Tests for the patch_embed module."""

import pytest
import torch
import torch.nn as nn
from yoke.models.vit.patch_embed import ParallelVarPatchEmbed
from yoke.models.vit.patch_embed import SwinEmbedding
from yoke.models.vit.patch_embed import _get_conv2d_weights, _get_conv2d_biases


def test_get_conv2d_weights() -> None:
    """Test _get_conv2d_weights function for correct shape and initialization."""
    in_channels, out_channels, kernel_size = 1, 64, (16, 16)
    weights = _get_conv2d_weights(in_channels, out_channels, kernel_size)

    assert weights.shape == (out_channels, in_channels, *kernel_size), (
        f"Expected shape {(out_channels, in_channels, *kernel_size)}",
        f"got {weights.shape}",
    )

    assert torch.is_tensor(weights), "Output should be a torch.Tensor"


def test_get_conv2d_biases() -> None:
    """Test _get_conv2d_biases function for correct shape and initialization."""
    out_channels = 64
    biases = _get_conv2d_biases(out_channels)

    assert biases.shape == (out_channels,), (
        f"Expected shape {(out_channels,)}, got {biases.shape}"
    )
    assert torch.is_tensor(biases), "Output should be a torch.Tensor"


def test_initialization() -> None:
    """Test initialization of ParallelVarPatchEmbed."""
    embedder = ParallelVarPatchEmbed(
        max_vars=3,
        img_size=(128, 128),
        patch_size=(16, 16),
        embed_dim=64,
        norm_layer=nn.LayerNorm,
    )

    assert embedder.max_vars == 3
    assert embedder.img_size == (128, 128)
    assert embedder.patch_size == (16, 16)
    assert embedder.embed_dim == 64
    assert embedder.num_patches == 64  # 128/16 * 128/16
    assert isinstance(embedder.norm, nn.LayerNorm)


def test_invalid_image_patch_size() -> None:
    """Test error raising for invalid image and patch size combinations."""
    with pytest.raises(
        AssertionError, match="Image height not divisible by patch height!!!"
    ):
        ParallelVarPatchEmbed(img_size=(130, 128), patch_size=(16, 16))

    with pytest.raises(
        AssertionError, match="Image width not divisible by patch width!!!"
    ):
        ParallelVarPatchEmbed(img_size=(128, 130), patch_size=(16, 16))


def test_forward_shape() -> None:
    """Test the forward pass and output shape."""
    embedder = ParallelVarPatchEmbed(
        max_vars=3,
        img_size=(128, 128),
        patch_size=(16, 16),
        embed_dim=64,
    )

    B, C, H, W = 2, 3, 128, 128
    x = torch.randn(B, C, H, W)
    in_vars = torch.tensor([0, 1, 2])

    output = embedder(x, in_vars)

    assert output.shape == (B, C, embedder.num_patches, embedder.embed_dim), (
        f"Expected shape {(B, C, embedder.num_patches, embedder.embed_dim)},"
        f"got {output.shape}"
    )


def test_reset_parameters() -> None:
    """Test that reset_parameters initializes weights and biases correctly."""
    embedder = ParallelVarPatchEmbed(max_vars=3)
    embedder.reset_parameters()

    for idx in range(embedder.max_vars):
        weights = embedder.proj_weights[idx]
        biases = embedder.proj_biases[idx]
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weights)
        bound = 1 / (fan_in**0.5) if fan_in > 0 else 0
        assert torch.all(weights <= bound).item() is True
        assert torch.all(weights >= -bound).item() is True
        assert torch.all(biases <= bound).item() is True
        assert torch.all(biases >= -bound).item() is True


def test_swin_initialization() -> None:
    """Test initialization of SwinEmbedding."""
    embedder = SwinEmbedding(
        num_vars=3,
        img_size=(128, 128),
        patch_size=(16, 16),
        embed_dim=64,
        norm_layer=nn.LayerNorm,
    )

    assert embedder.num_vars == 3
    assert embedder.img_size == (128, 128)
    assert embedder.patch_size == (16, 16)
    assert embedder.embed_dim == 64
    assert embedder.num_patches == 64  # 128/16 * 128/16
    assert isinstance(embedder.norm, nn.LayerNorm)
    assert isinstance(embedder.linear_embedding, nn.Conv2d)


def test_swin_invalid_image_patch_size() -> None:
    """Test error raising for invalid image and patch size combinations."""
    with pytest.raises(
        AssertionError, match="Image height not divisible by patch height!!!"
    ):
        SwinEmbedding(img_size=(130, 128), patch_size=(16, 16))

    with pytest.raises(
        AssertionError, match="Image width not divisible by patch widht!!!"
    ):
        SwinEmbedding(img_size=(128, 130), patch_size=(16, 16))


def test_swin_forward_shape() -> None:
    """Test the forward pass and output shape of SwinEmbedding."""
    embedder = SwinEmbedding(
        num_vars=3,
        img_size=(128, 128),
        patch_size=(16, 16),
        embed_dim=64,
    )

    B, C, H, W = 2, 3, 128, 128
    x = torch.randn(B, C, H, W)

    output = embedder(x)
    assert output.shape == (B, embedder.num_patches, embedder.embed_dim), (
        f"Expected shape {(B, embedder.num_patches, embedder.embed_dim)},"
        f"got {output.shape}"
    )


def test_swin_linear_embedding_weights() -> None:
    """Test that linear embedding layer initializes correctly."""
    embedder = SwinEmbedding(num_vars=3, embed_dim=64, patch_size=(16, 16))

    conv = embedder.linear_embedding

    assert isinstance(conv, nn.Conv2d), (
        "linear_embedding should be an nn.Conv2d instance"
    )

    assert conv.kernel_size == (
        16,
        16,
    ), f"Expected kernel_size (16, 16), got {conv.kernel_size}"

    assert conv.stride == (16, 16), f"Expected stride (16, 16), got {conv.stride}"
