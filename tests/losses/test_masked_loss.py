"""Test masked loss module."""

import pytest

import torch

from yoke.losses.masked_loss import MaskedLossMultiplicative


@pytest.fixture
def masked_loss() -> MaskedLossMultiplicative:
    """Fixture for masked loss tests."""
    return MaskedLossMultiplicative(torch.nn.MSELoss(reduction="none"))


def test_masked_loss_init(masked_loss: MaskedLossMultiplicative) -> None:
    """Test initialization."""
    assert isinstance(masked_loss, MaskedLossMultiplicative)


def test_loderunner_forward(masked_loss: MaskedLossMultiplicative) -> None:
    """Test forward method."""
    input = torch.randn(2, 3, 1120, 800)  # Batch size of 2, 3 channels, image size
    target = torch.randn(2, 3, 1120, 800)
    masked_loss(input=input, target=target)
