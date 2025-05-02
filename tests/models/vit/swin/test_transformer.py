"""Tests for th Swin transformer classes."""

import pytest
import torch
from torch import nn
from yoke.models.vit.swin.transformer import Swin, SwinV2


@pytest.fixture
def dummy_input() -> torch.Tensor:
    """Create a dummy input tensor for testing."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.randn(2, 3, 1120, 800).to(device)


@pytest.fixture
def swin_model() -> nn.Module:
    """Create an instance of the Swin model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return Swin(
        input_channels=3,
        img_size=(1120, 800),
        patch_size=(10, 10),
        block_structure=(1, 1, 3, 1),
        emb_size=96,
        emb_factor=2,
        num_heads=8,
        window_sizes=[(8, 8), (8, 8), (4, 4), (2, 2)],
        patch_merge_scales=[(2, 2), (2, 2), (2, 2)],
        num_output_classes=5,
        verbose=False,
    ).to(device)


@pytest.fixture
def swin_v2_model() -> nn.Module:
    """Create an instance of the SwinV2 model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SwinV2(
        input_channels=3,
        img_size=(1120, 800),
        patch_size=(10, 10),
        block_structure=(1, 1, 3, 1),
        emb_size=96,
        emb_factor=2,
        num_heads=8,
        window_sizes=[(8, 8), (8, 8), (4, 4), (2, 2)],
        patch_merge_scales=[(2, 2), (2, 2), (2, 2)],
        num_output_classes=5,
        verbose=False,
    ).to(device)


def test_swin_initialization(swin_model: Swin) -> None:
    """Test that the Swin model initializes correctly."""
    assert isinstance(swin_model, Swin)
    assert swin_model.input_channels == 3
    assert swin_model.img_size == (1120, 800)
    assert swin_model.num_output_classes == 5
    assert swin_model.emb_size == 96
    assert len(swin_model.stage1) == 1
    assert len(swin_model.stage2) == 1
    assert len(swin_model.stage3) == 3
    assert len(swin_model.stage4) == 1


def test_swin_forward(swin_model: Swin, dummy_input: torch.Tensor) -> None:
    """Test the forward pass of the Swin model."""
    output = swin_model(dummy_input)
    assert output.shape == (2, swin_model.num_output_classes)
    assert isinstance(output, torch.Tensor)


def test_swin_v2_initialization(swin_v2_model: SwinV2) -> None:
    """Test that the SwinV2 model initializes correctly."""
    assert isinstance(swin_v2_model, SwinV2)
    assert swin_v2_model.input_channels == 3
    assert swin_v2_model.img_size == (1120, 800)
    assert swin_v2_model.num_output_classes == 5
    assert swin_v2_model.emb_size == 96
    assert len(swin_v2_model.stage1) > 0
    assert len(swin_v2_model.stage2) > 0
    assert len(swin_v2_model.stage3) > 0
    assert len(swin_v2_model.stage4) > 0


def test_swin_v2_forward(swin_v2_model: SwinV2, dummy_input: torch.Tensor) -> None:
    """Test the forward pass of the SwinV2 model."""
    output = swin_v2_model(dummy_input)
    assert output.shape == (2, swin_v2_model.num_output_classes)
    assert isinstance(output, torch.Tensor)


def test_swin_embedding_shapes(swin_model: Swin, dummy_input: torch.Tensor) -> None:
    """Test the intermediate embedding shapes in Swin."""
    embedding = swin_model.Embedding(dummy_input)
    grid_size = swin_model.patch_grid_size
    assert embedding.shape == (2, grid_size[0] * grid_size[1], swin_model.emb_size)


def test_swin_v2_embedding_shapes(
    swin_v2_model: SwinV2, dummy_input: torch.Tensor
) -> None:
    """Test the intermediate embedding shapes in SwinV2."""
    embedding = swin_v2_model.Embedding(dummy_input)
    grid_size = swin_v2_model.patch_grid_size
    assert embedding.shape == (2, grid_size[0] * grid_size[1], swin_v2_model.emb_size)
