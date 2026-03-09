"""Tests for the LSC policy datastep."""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import pytest
from _pytest.monkeypatch import MonkeyPatch

from yoke.utils.training.datastep.lsc_policy import (
    train_lsc_policy_datastep,
    eval_lsc_policy_datastep,
)


class DummyDistribution:
    """Dummy distribution with a mean attribute.

    Attributes:
        mean (torch.Tensor): The predicted mean tensor.
    """

    def __init__(self, mean: torch.Tensor) -> None:
        """Initialize with a mean tensor."""
        self.mean = mean


class DummyModel(nn.Module):
    """Dummy model that sums inputs and returns a DummyDistribution."""

    def __init__(self) -> None:
        """Initialize the dummy model."""
        super().__init__()
        # ensure there's at least one parameter for the optimizer
        self.dummy_param = nn.Parameter(torch.tensor(1.0))

    def forward(
        self, state_y: torch.Tensor, stateH: torch.Tensor, targetH: torch.Tensor
    ) -> DummyDistribution:
        """Forward that sums state_y, stateH, and targetH."""
        sum_tensor = state_y + stateH + targetH
        pred = sum_tensor + self.dummy_param

        return DummyDistribution(pred)


def test_train_lsc_policy_datastep_single_rank(monkeypatch: MonkeyPatch) -> None:
    """Test train step with single rank and CPU device.

    Args:
        monkeypatch: Fixture for patching dist.all_gather.
    """
    batch_size = 5
    feature_size = 3
    state_y = torch.ones(batch_size, feature_size)
    stateH = torch.ones(batch_size, feature_size) * 2
    targetH = torch.ones(batch_size, feature_size) * 3
    x_true = torch.ones(batch_size, feature_size) * 6
    data = (state_y, stateH, targetH, x_true)

    monkeypatch.setattr(dist, "all_gather", lambda outs, t: outs.__setitem__(0, t))
    model = DummyModel()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss(reduction="none")
    device = torch.device("cpu")

    x_ret, pred_mean, all_losses = train_lsc_policy_datastep(
        data, model, optimizer, loss_fn, device, rank=0, world_size=1
    )

    assert model.training is True
    assert torch.allclose(x_ret, x_true)
    # sum 1+2+3=6, plus dummy_param (1.0) => 7
    expected = torch.full((batch_size, feature_size), 7.0)
    assert torch.allclose(pred_mean, expected)
    assert all_losses.shape == (batch_size,)
    # per-sample MSE loss = (7-6)^2 averaged over features = 1.0
    assert torch.allclose(all_losses, torch.ones(batch_size))


def test_eval_lsc_policy_datastep_single_rank(monkeypatch: MonkeyPatch) -> None:
    """Test eval step with single rank and CPU device.

    Args:
        monkeypatch: Fixture for patching dist.all_gather.
    """
    batch_size = 4
    feature_size = 2
    state_y = torch.full((batch_size, feature_size), 2.0)
    stateH = torch.full((batch_size, feature_size), 3.0)
    targetH = torch.full((batch_size, feature_size), 4.0)
    x_true = torch.full((batch_size, feature_size), 9.0)
    data = (state_y, stateH, targetH, x_true)

    monkeypatch.setattr(dist, "all_gather", lambda outs, t: outs.__setitem__(0, t))
    model = DummyModel()
    loss_fn = nn.MSELoss(reduction="none")
    device = torch.device("cpu")

    x_ret, pred_mean, all_losses = eval_lsc_policy_datastep(
        data, model, loss_fn, device, rank=0, world_size=1
    )

    assert model.training is False
    assert torch.allclose(x_ret, x_true)
    # sum 2+3+4=9, plus dummy_param (1.0) => 10
    expected = torch.full((batch_size, feature_size), 10.0)
    assert torch.allclose(pred_mean, expected)
    assert all_losses.shape == (batch_size,)
    # per-sample MSE loss = (10-9)^2 averaged over features = 1.0
    assert torch.allclose(all_losses, torch.ones(batch_size))


@pytest.mark.parametrize("rank,world_size", [(0, 2), (1, 2)])
def test_train_lsc_policy_datastep_multi_rank(
    monkeypatch: MonkeyPatch, rank: int, world_size: int
) -> None:
    """Test train step with multiple ranks.

    Args:
        monkeypatch: Fixture for patching dist.all_gather.
        rank: Process rank.
        world_size: Total number of processes.
    """
    batch_size = 3
    feature_size = 1
    state_y = torch.ones(batch_size, feature_size)
    stateH = torch.zeros(batch_size, feature_size)
    targetH = torch.zeros(batch_size, feature_size)
    x_true = torch.ones(batch_size, feature_size)
    data = (state_y, stateH, targetH, x_true)

    def stub_all_gather(outs: list, t: torch.Tensor) -> None:
        for i in range(len(outs)):
            outs[i] = t + i

    monkeypatch.setattr(dist, "all_gather", stub_all_gather)
    model = DummyModel()
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss(reduction="none")
    device = torch.device("cpu")

    x_ret, pred_mean_val, all_losses = train_lsc_policy_datastep(
        data, model, optimizer, loss_fn, device, rank, world_size
    )

    assert torch.allclose(x_ret, x_true)
    # sum 1+0+0=1, plus dummy_param (1.0) => 2
    expected_pm = torch.full((batch_size, feature_size), 2.0)
    assert torch.allclose(pred_mean_val, expected_pm)
    if rank == 0:
        assert all_losses.shape == (world_size * batch_size,)
        # per-sample loss before gather is (2-1)^2 = 1.0
        # stub_all_gather yields [ones, twos]
        assert torch.equal(all_losses[:batch_size], torch.ones(batch_size))
        assert torch.equal(all_losses[batch_size:], torch.full((batch_size,), 2.0))
    else:
        assert all_losses is None


@pytest.mark.parametrize("rank,world_size", [(0, 2), (1, 2)])
def test_eval_lsc_policy_datastep_multi_rank(
    monkeypatch: MonkeyPatch, rank: int, world_size: int
) -> None:
    """Test eval step with multiple ranks.

    Args:
        monkeypatch: Fixture for patching dist.all_gather.
        rank: Process rank.
        world_size: Total number of processes.
    """
    batch_size = 2
    feature_size = 2
    state_y = torch.zeros(batch_size, feature_size)
    stateH = torch.ones(batch_size, feature_size)
    targetH = torch.zeros(batch_size, feature_size)
    x_true = torch.ones(batch_size, feature_size)
    data = (state_y, stateH, targetH, x_true)

    def stub_all_gather(outs: list, t: torch.Tensor) -> None:
        for i in range(len(outs)):
            outs[i] = t * (i + 1)

    monkeypatch.setattr(dist, "all_gather", stub_all_gather)
    model = DummyModel()
    loss_fn = nn.MSELoss(reduction="none")
    device = torch.device("cpu")

    x_ret, pred_mean_val, all_losses = eval_lsc_policy_datastep(
        data, model, loss_fn, device, rank, world_size
    )
    # sum 0+1+0=1, plus dummy_param (1.0) => 2
    expected_pm = torch.full((batch_size, feature_size), 2.0)
    assert torch.allclose(x_ret, x_true)

    assert torch.allclose(pred_mean_val, expected_pm)
    if rank == 0:
        assert all_losses.shape == (world_size * batch_size,)
        # stub_all_gather yields [ones, twos]
        expected_losses = torch.tensor([1.0] * batch_size + [2.0] * batch_size)
        assert torch.equal(all_losses, expected_losses)
    else:
        assert all_losses is None
