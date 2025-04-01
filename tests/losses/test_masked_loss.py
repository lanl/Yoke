"""Test masked loss module."""

import pytest

import torch

from yoke.losses.masked_loss import CroppedLoss2D, MaskedLossMultiplicative


IM_SIZE = (1120, 400)


@pytest.fixture
def cropped_loss() -> CroppedLoss2D:
    """Fixture for cropped loss tests."""
    return CroppedLoss2D(torch.nn.MSELoss(reduction="none"), crop=(0, 0, *IM_SIZE))


def test_cropped_loss_init(cropped_loss: CroppedLoss2D) -> None:
    """Test initialization."""
    assert isinstance(cropped_loss, CroppedLoss2D)


def test_cropped_loss_forward(cropped_loss: CroppedLoss2D) -> None:
    """Test forward method."""
    input = torch.randn(2, 3, *IM_SIZE)  # Batch size of 2, 3 channels, image size
    target = torch.randn(2, 3, *IM_SIZE)
    cropped_loss(input=input, target=target)


@pytest.fixture
def masked_loss() -> MaskedLossMultiplicative:
    """Fixture for masked loss tests."""
    return MaskedLossMultiplicative(torch.nn.MSELoss(reduction="none"))


def test_masked_loss_init(masked_loss: MaskedLossMultiplicative) -> None:
    """Test initialization."""
    assert isinstance(masked_loss, MaskedLossMultiplicative)


def test_masked_loss_forward(masked_loss: MaskedLossMultiplicative) -> None:
    """Test forward method."""
    input = torch.randn(2, 3, *IM_SIZE)  # Batch size of 2, 3 channels, image size
    target = torch.randn(2, 3, *IM_SIZE)
    masked_loss(input=input, target=target)
