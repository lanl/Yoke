"""Tests for the NormalizedMSELoss class."""

import pytest
import torch
from yoke.losses.NormMSE import NormalizedMSELoss


@pytest.fixture
def norm_mse() -> NormalizedMSELoss:
    """Fixture for NormalizedMSELoss."""
    return NormalizedMSELoss()


def test_norm_mse_loss_zero(norm_mse: NormalizedMSELoss) -> None:
    """Test the NormalizedMSELoss with zero input and target."""
    inp = torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]])
    target = torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]])
    loss = norm_mse(inp, target)
    assert torch.all(loss == 0.0)


def test_norm_mse_loss_positive(norm_mse: NormalizedMSELoss) -> None:
    """Test the NormalizedMSELoss with positive input and target."""
    inp = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=torch.float32)
    target = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=torch.float32)
    loss = norm_mse(inp, target)
    assert torch.all(loss == 0.0)


def test_norm_mse_loss_non_zero(norm_mse: NormalizedMSELoss) -> None:
    """Test the NormalizedMSELoss with non-zero input and target."""
    inp = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    target = torch.tensor([[[[4.0, 5.0], [6.0, 7.0]]]])
    loss = norm_mse(inp, target)
    expected_loss = torch.mean(
        (
            (inp - target.mean(dim=(0, 2, 3), keepdim=True))
            / (target.std(dim=(0, 2, 3), keepdim=True) + norm_mse.eps)
            - (target - target.mean(dim=(0, 2, 3), keepdim=True))
            / (target.std(dim=(0, 2, 3), keepdim=True) + norm_mse.eps)
        )
        ** 2,
        dim=(0, 2, 3),
    )
    assert torch.all(loss == expected_loss)


def test_norm_mse_loss_negative(norm_mse: NormalizedMSELoss) -> None:
    """Test the NormalizedMSELoss with negative input and target."""
    inp = torch.tensor([[[[-1.0, -2.0], [-3.0, -4.0]]]])
    target = torch.tensor([[[[-4.0, -5.0], [-6.0, -7.0]]]])
    loss = norm_mse(inp, target)
    expected_loss = torch.mean(
        (
            (inp - target.mean(dim=(0, 2, 3), keepdim=True))
            / (target.std(dim=(0, 2, 3), keepdim=True) + norm_mse.eps)
            - (target - target.mean(dim=(0, 2, 3), keepdim=True))
            / (target.std(dim=(0, 2, 3), keepdim=True) + norm_mse.eps)
        )
        ** 2,
        dim=(0, 2, 3),
    )
    assert torch.all(loss == expected_loss)


def test_norm_mse_loss_mean_reduction() -> None:
    """Test the mean reduction of the NormalizedMSELoss."""
    norm_mse = NormalizedMSELoss(reduction="mean")
    inp = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    target = torch.tensor([[[[4.0, 5.0], [6.0, 7.0]]]])
    loss = norm_mse(inp, target)
    expected_loss = torch.mean(
        (
            (inp - target.mean(dim=(0, 2, 3), keepdim=True))
            / (target.std(dim=(0, 2, 3), keepdim=True) + norm_mse.eps)
            - (target - target.mean(dim=(0, 2, 3), keepdim=True))
            / (target.std(dim=(0, 2, 3), keepdim=True) + norm_mse.eps)
        )
        ** 2
    ).mean()
    assert torch.all(loss == expected_loss)


def test_norm_mse_loss_sum_reduction() -> None:
    """Test the sum reduction of the NormalizedMSELoss."""
    norm_mse = NormalizedMSELoss(reduction="sum")
    inp = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    target = torch.tensor([[[[4.0, 5.0], [6.0, 7.0]]]])
    loss = norm_mse(inp, target)
    expected_loss = torch.sum(
        (
            (inp - target.mean(dim=(0, 2, 3), keepdim=True))
            / (target.std(dim=(0, 2, 3), keepdim=True) + norm_mse.eps)
            - (target - target.mean(dim=(0, 2, 3), keepdim=True))
            / (target.std(dim=(0, 2, 3), keepdim=True) + norm_mse.eps)
        )
        ** 2
    ).sum()
    assert loss == expected_loss


def test_norm_mse_loss_different_shapes(norm_mse: NormalizedMSELoss) -> None:
    """Test the NormalizedMSELoss with different input and target shapes."""
    inp = torch.tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]])
    target = torch.tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]])
    loss = norm_mse(inp, target)
    assert torch.all(loss == 0.0)


def test_norm_mse_loss_batch_size(norm_mse: NormalizedMSELoss) -> None:
    """Test the NormalizedMSELoss with different batch sizes."""
    inp = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]], [[[1.0, 2.0], [3.0, 4.0]]]])
    target = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]], [[[1.0, 2.0], [3.0, 4.0]]]])
    loss = norm_mse(inp, target)
    assert torch.all(loss == 0.0)


def test_norm_mse_loss_different_eps() -> None:
    """Test the NormalizedMSELoss with different eps values."""
    norm_mse = NormalizedMSELoss(eps=1e-5)
    inp = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    target = torch.tensor([[[[4.0, 5.0], [6.0, 7.0]]]])
    loss = norm_mse(inp, target)
    expected_loss = (
        (inp - target.mean(dim=(0, 2, 3), keepdim=True))
        / (target.std(dim=(0, 2, 3), keepdim=True) + 1e-5)
        - (target - target.mean(dim=(0, 2, 3), keepdim=True))
        / (target.std(dim=(0, 2, 3), keepdim=True) + 1e-5)
    ) ** 2
    assert torch.all(loss == expected_loss)
