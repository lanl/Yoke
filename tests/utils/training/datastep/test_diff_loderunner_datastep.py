"""Tests for the DiffusionLodeRunner training/evaluation datastep utilities."""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import pytest

from yoke.utils.training.datastep.diff_loderunner import (
    train_diffusion_loderunner_datastep,
    train_DDP_diffusion_loderunner_datastep,
    eval_diffusion_loderunner_datastep,
    eval_DDP_diffusion_loderunner_datastep,
)


class DummyDiffusionModel(nn.Module):
    """Dummy diffusion model returning ``y_tau + param`` and recording call kwargs."""

    def __init__(self) -> None:
        """Initialize with a single trainable parameter and a call recorder."""
        super().__init__()
        self.param = nn.Parameter(torch.tensor(1.0))
        self.last_kwargs: dict = {}

    def forward(
        self,
        x: torch.Tensor,
        y_tau: torch.Tensor,
        in_vars: torch.Tensor,
        out_vars: torch.Tensor,
        lead_times: torch.Tensor,
        diffusion_time: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``y_tau + param`` while recording the keyword arguments used.

        Args:
            x (torch.Tensor): Conditioning input (unused in output).
            y_tau (torch.Tensor): Noised target; the returned prediction is
                ``y_tau + param``.
            in_vars (torch.Tensor): Input variable indices.
            out_vars (torch.Tensor): Output variable indices.
            lead_times (torch.Tensor): Lead time values.
            diffusion_time (torch.Tensor): Diffusion time values.

        Returns:
            torch.Tensor: ``y_tau`` incremented by the model parameter.
        """
        self.last_kwargs = {
            "in_vars": in_vars,
            "out_vars": out_vars,
            "lead_times": lead_times,
            "diffusion_time": diffusion_time,
        }
        return y_tau + self.param


@pytest.fixture(autouse=True)
def patch_all_gather(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch ``dist.all_gather`` to copy the tensor into each output slot."""

    def fake_all_gather(output_list: list, tensor: torch.Tensor) -> None:
        for i in range(len(output_list)):
            output_list[i].copy_(tensor)

    monkeypatch.setattr(dist, "all_gather", fake_all_gather)


@pytest.fixture
def loss_fn() -> nn.Module:
    """Return an elementwise MSE loss."""
    return nn.MSELoss(reduction="none")


@pytest.fixture
def device() -> torch.device:
    """Return CPU device."""
    return torch.device("cpu")


def _make_batch(B: int = 2, C: int = 8, H: int = 2, W: int = 2) -> tuple:
    """Build a diffusion batch ``(x, y_tau, noise, lead_times, tau)``.

    Args:
        B (int): Batch size.
        C (int): Channel count.
        H (int): Image height.
        W (int): Image width.

    Returns:
        tuple: The five-element diffusion data batch.
    """
    x = torch.zeros((B, C, H, W))
    y_tau = torch.zeros((B, C, H, W))
    noise = torch.ones((B, C, H, W))
    lead_times = torch.ones((B,))
    tau = torch.full((B,), 0.5)
    return x, y_tau, noise, lead_times, tau


def test_train_diffusion_loderunner_datastep(
    device: torch.device, loss_fn: nn.Module
) -> None:
    """Training step returns ground-truth noise, prediction, and per-sample loss."""
    model = DummyDiffusionModel()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    B, C, H, W = 2, 8, 2, 2
    data = _make_batch(B, C, H, W)
    in_vars = torch.arange(C)
    out_vars = torch.arange(C)

    noise, noise_pred, per_loss = train_diffusion_loderunner_datastep(
        data, model, optimizer, loss_fn, device, in_vars, out_vars
    )

    # noise returned unmodified; prediction is y_tau + 1.0 (zeros + 1)
    assert torch.equal(noise, torch.ones((B, C, H, W)))
    assert torch.equal(noise_pred, torch.ones((B, C, H, W)))
    # MSE((y_tau+1) - noise) = 0 since both are ones
    assert per_loss.shape == (B,)
    assert torch.allclose(per_loss, torch.zeros(B))
    # datastep hardcodes the 8-variable index tensor for in/out vars
    assert torch.equal(model.last_kwargs["in_vars"], torch.arange(8))
    assert torch.equal(model.last_kwargs["out_vars"], torch.arange(8))


@pytest.mark.parametrize("rank", [0, 1])
def test_train_DDP_diffusion_loderunner_datastep(
    rank: int, device: torch.device, loss_fn: nn.Module
) -> None:
    """DDP training step gathers losses and returns them only on rank 0."""
    model = DummyDiffusionModel()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    B, C, H, W = 2, 8, 2, 2
    data = _make_batch(B, C, H, W)
    in_vars = torch.arange(C)
    out_vars = torch.arange(C)
    world_size = 3

    noise, noise_pred, all_losses = train_DDP_diffusion_loderunner_datastep(
        data, model, optimizer, loss_fn, device, rank, world_size, in_vars, out_vars
    )

    assert torch.equal(noise, torch.ones((B, C, H, W)))
    assert torch.equal(noise_pred, torch.ones((B, C, H, W)))
    if rank == 0:
        assert all_losses is not None
        assert all_losses.shape == (world_size * B,)
    else:
        assert all_losses is None


def test_eval_diffusion_loderunner_datastep(
    device: torch.device, loss_fn: nn.Module
) -> None:
    """Eval step returns noise, prediction, and per-sample loss without grad."""
    model = DummyDiffusionModel()
    B, C, H, W = 2, 8, 2, 2
    data = _make_batch(B, C, H, W)
    in_vars = torch.arange(C)
    out_vars = torch.arange(C)

    noise, noise_pred, per_loss = eval_diffusion_loderunner_datastep(
        data, model, loss_fn, device, in_vars, out_vars
    )

    assert torch.equal(noise, torch.ones((B, C, H, W)))
    assert torch.equal(noise_pred, torch.ones((B, C, H, W)))
    assert per_loss.shape == (B,)
    assert torch.allclose(per_loss, torch.zeros(B))
    # eval passes the provided in/out vars through to the model
    assert torch.equal(model.last_kwargs["in_vars"], in_vars)
    assert torch.equal(model.last_kwargs["out_vars"], out_vars)
    # no_grad context: prediction should not require grad
    assert not noise_pred.requires_grad


@pytest.mark.parametrize("rank", [0, 1])
def test_eval_DDP_diffusion_loderunner_datastep(
    rank: int, device: torch.device, loss_fn: nn.Module
) -> None:
    """DDP eval step gathers losses and returns them only on rank 0."""
    model = DummyDiffusionModel()
    B, C, H, W = 2, 8, 2, 2
    data = _make_batch(B, C, H, W)
    in_vars = torch.arange(C)
    out_vars = torch.arange(C)
    world_size = 4

    noise, noise_pred, all_losses = eval_DDP_diffusion_loderunner_datastep(
        data, model, loss_fn, device, rank, world_size, in_vars, out_vars
    )

    assert torch.equal(noise, torch.ones((B, C, H, W)))
    assert torch.equal(noise_pred, torch.ones((B, C, H, W)))
    if rank == 0:
        assert all_losses is not None
        assert all_losses.shape == (world_size * B,)
    else:
        assert all_losses is None
