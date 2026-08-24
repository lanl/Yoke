"""Tests for the LodeRunner training datastep utilities."""

# test_loderunner_datastep.py

import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import pytest

from yoke.utils.training.datastep.loderunner import (
    train_loderunner_datastep,
    train_scheduled_loderunner_datastep,
    train_DDP_loderunner_datastep,
    eval_loderunner_datastep,
    eval_scheduled_loderunner_datastep,
    eval_DDP_loderunner_datastep,
    eval_loderunner_datastep_cylex,
    train_DDP_loderunner_datastep_cylex,
    eval_DDP_loderunner_datastep_cylex,
    train_DDP_loderunner_2frame_datastep,
    eval_DDP_loderunner_2frame_datastep,
)


class DummyModel(nn.Module):
    """A dummy model that adds 1.0 to its input and holds one parameter."""

    def __init__(self) -> None:
        """Initialize the DummyModel with a single parameter."""
        super().__init__()
        self.param = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        x: torch.Tensor,
        in_vars: torch.Tensor,
        out_vars: torch.Tensor,
        Dt: torch.Tensor,
    ) -> torch.Tensor:
        """Add 1.0 to the input tensor x and return the result.

        Args:
            x (torch.Tensor): Input tensor.
            in_vars (torch.Tensor): Input variables (unused).
            out_vars (torch.Tensor): Output variables (unused).
            Dt (torch.Tensor): Time step tensor (unused).

        Returns:
            torch.Tensor: The input tensor x incremented by 1.0.
        """
        return x + self.param


class DummyTwoFrameModel(nn.Module):
    """Dummy 2-in/1-out model: averages the two input frames then adds 1.0."""

    def __init__(self) -> None:
        """Initialize with a single parameter."""
        super().__init__()
        self.param = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        x: torch.Tensor,
        in_vars: torch.Tensor,
        out_vars: torch.Tensor,
        lead_times: torch.Tensor,
    ) -> torch.Tensor:
        """Reduce the (B, 2, C, H, W) input over the time axis and add 1.0.

        Args:
            x (torch.Tensor): Input of shape (B, 2, C, H, W).
            in_vars (torch.Tensor): Input variables (unused).
            out_vars (torch.Tensor): Output variables (unused).
            lead_times (torch.Tensor): (B, 2) lead-times (unused).

        Returns:
            torch.Tensor: Prediction of shape (B, C, H, W).
        """
        return x.mean(dim=1) + self.param


@pytest.fixture(autouse=True)
def patch_ddp_and_random(monkeypatch: pytest.MonkeyPatch) -> None:
    """Seed Python RNG and patch dist.all_gather for DDP tests."""
    random.seed(0)

    def fake_all_gather(output_list: list, tensor: torch.Tensor) -> None:
        for i in range(len(output_list)):
            output_list[i].copy_(tensor)

    monkeypatch.setattr(dist, "all_gather", fake_all_gather)


@pytest.fixture
def loss_fn() -> nn.Module:
    """Return an elementwise MSE loss for testing."""
    return nn.MSELoss(reduction="none")


@pytest.fixture
def device() -> torch.device:
    """Return CPU device."""
    return torch.device("cpu")


def test_train_loderunner_datastep_basic(
    device: torch.device, loss_fn: nn.Module
) -> None:
    """Test train_loderunner_datastep returns expected shapes and values."""
    model = DummyModel()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    B, C, H, W = 2, 1, 2, 2
    start = torch.zeros((B, C, H, W))
    end = torch.zeros((B, C, H, W))
    Dt = torch.ones((B, 1))
    end_img, pred_img, per_loss = train_loderunner_datastep(
        (start, end, Dt), model, optimizer, loss_fn, device, channel_map=[0]
    )
    # end_img is returned unmodified
    assert torch.equal(end_img, end)
    # pred_img == start + 1.0
    assert torch.equal(pred_img, start + 1.0)
    # per-sample loss: MSE((start+1)-end)=1.0 averaged over dims
    assert per_loss.shape == (B,)
    assert torch.allclose(per_loss, torch.ones(B))


@pytest.mark.parametrize("p", [1.0, 0.0])
def test_train_scheduled_loderunner_datastep_branches(
    p: float, device: torch.device, loss_fn: nn.Module
) -> None:
    """Test scheduled training with both branches of scheduled_prob."""
    model = DummyModel()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    # Build img_seq shape [B, S, C, H, W] with distinct values
    B, S, C, H, W = 2, 3, 1, 2, 2
    base = torch.arange(S, dtype=torch.float32)
    # shape (S,) -> (S, C, H, W)
    slices = [b.repeat(C, H, W) for b in base]
    seq = torch.stack(slices, dim=0).unsqueeze(0).repeat(B, 1, 1, 1, 1)
    Dt = torch.ones((B, 1))
    img_gt, pred_seq, per_loss = train_scheduled_loderunner_datastep(
        (seq, Dt), model, optimizer, loss_fn, device, scheduled_prob=p
    )
    # img_gt == seq[:,1:]
    assert torch.equal(img_gt, seq[:, 1:])
    # pred_seq shape [B, S-1, C, H, W]
    assert pred_seq.shape == (B, S - 1, C, H, W)
    # Because model(x)=x+1, for both p=0 and p=1 pred_seq == seq[:,:-1]+1
    expected = seq[:, :-1] + 1.0
    assert torch.equal(pred_seq, expected)
    # per_loss zeros because expected==seq[:,1:]
    assert per_loss.shape == (B,)
    assert torch.allclose(per_loss, torch.zeros(B))


@pytest.mark.parametrize("rank", [0, 1])
def test_train_DDP_loderunner_datastep(
    rank: int, device: torch.device, loss_fn: nn.Module
) -> None:
    """Test DDP training step returns all_losses only on rank 0."""
    model = DummyModel()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    B, C, H, W = 2, 1, 2, 2
    start = torch.zeros((B, C, H, W))
    end = torch.zeros((B, C, H, W))
    Dt = torch.ones((B, 1))
    world_size = 3
    end_img, pred_img, all_losses = train_DDP_loderunner_datastep(
        (start, end, Dt),
        model,
        optimizer,
        loss_fn,
        device,
        rank,
        world_size,
    )
    assert torch.equal(end_img, end)
    assert torch.equal(pred_img, start + 1.0)
    if rank == 0:
        assert all_losses is not None
        assert all_losses.shape == (world_size * B,)
    else:
        assert all_losses is None


@pytest.mark.parametrize("rank", [0, 1])
def test_train_DDP_loderunner_2frame_datastep(
    rank: int, device: torch.device, loss_fn: nn.Module
) -> None:
    """Test the 2-frame DDP training step shapes and rank-0 loss gathering."""
    model = DummyTwoFrameModel()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    B, T, C, H, W = 2, 2, 1, 2, 2
    input_frames = torch.zeros((B, T, C, H, W))
    target = torch.zeros((B, C, H, W))
    lead_times = torch.ones((B, 2))
    world_size = 3

    end_img, pred_img, all_losses = train_DDP_loderunner_2frame_datastep(
        (input_frames, target, lead_times),
        model,
        optimizer,
        loss_fn,
        device,
        rank,
        world_size,
        channel_map=[0],
    )

    assert torch.equal(end_img, target)
    # mean over frames (all zeros) + 1.0
    assert torch.equal(pred_img, target + 1.0)
    if rank == 0:
        assert all_losses.shape == (world_size * B,)
    else:
        assert all_losses is None


def test_train_DDP_loderunner_2frame_datastep_grad_clip(
    device: torch.device, loss_fn: nn.Module, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test grad_clip invokes clip_grad_norm_ before the optimizer step."""
    model = DummyTwoFrameModel()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    B, T, C, H, W = 2, 2, 1, 2, 2
    input_frames = torch.ones((B, T, C, H, W))
    target = torch.zeros((B, C, H, W))
    lead_times = torch.ones((B, 2))

    called = {"n": 0}
    orig = torch.nn.utils.clip_grad_norm_

    def spy(*args: object, **kwargs: object) -> torch.Tensor:
        called["n"] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy)

    train_DDP_loderunner_2frame_datastep(
        (input_frames, target, lead_times),
        model,
        optimizer,
        loss_fn,
        device,
        rank=0,
        world_size=1,
        channel_map=[0],
        grad_clip=1.0,
    )

    assert called["n"] == 1


@pytest.mark.parametrize("rank", [0, 1])
def test_eval_DDP_loderunner_2frame_datastep(
    rank: int, device: torch.device, loss_fn: nn.Module
) -> None:
    """Test the 2-frame DDP eval step shapes and rank-0 loss gathering."""
    model = DummyTwoFrameModel()
    B, T, C, H, W = 2, 2, 1, 2, 2
    input_frames = torch.zeros((B, T, C, H, W))
    target = torch.zeros((B, C, H, W))
    lead_times = torch.ones((B, 2))
    world_size = 4

    end_img, pred_img, all_losses = eval_DDP_loderunner_2frame_datastep(
        (input_frames, target, lead_times),
        model,
        loss_fn,
        device,
        rank,
        world_size,
        channel_map=[0],
    )

    assert torch.equal(end_img, target)
    assert torch.equal(pred_img, target + 1.0)
    if rank == 0:
        assert all_losses.shape == (world_size * B,)
    else:
        assert all_losses is None



def test_eval_loderunner_datastep(device: torch.device, loss_fn: nn.Module) -> None:
    """Test eval_loderunner_datastep returns correct shapes and values."""
    model = DummyModel()
    B, C, H, W = 2, 1, 2, 2
    start = torch.zeros((B, C, H, W))
    end = torch.zeros((B, C, H, W))
    Dt = torch.ones((B, 1))
    end_img, pred_img, per_loss = eval_loderunner_datastep(
        (start, end, Dt), model, loss_fn, device, channel_map=[0]
    )
    assert torch.equal(end_img, end)
    assert torch.equal(pred_img, start + 1.0)
    assert per_loss.shape == (B,)
    assert torch.allclose(per_loss, torch.ones(B))


@pytest.mark.parametrize("p", [1.0, 0.0])
def test_eval_scheduled_loderunner_datastep(
    p: float, device: torch.device, loss_fn: nn.Module
) -> None:
    """Test eval_scheduled_loderunner_datastep for both scheduled_prob branches."""
    model = DummyModel()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    B, S, C, H, W = 2, 3, 1, 2, 2
    base = torch.arange(S, dtype=torch.float32)
    slices = [b.repeat(C, H, W) for b in base]
    seq = torch.stack(slices, dim=0).unsqueeze(0).repeat(B, 1, 1, 1, 1)
    Dt = torch.ones((B, 1))
    img_gt, pred_seq, per_loss = eval_scheduled_loderunner_datastep(
        (seq, Dt), model, optimizer, loss_fn, device, scheduled_prob=p
    )
    assert torch.equal(img_gt, seq[:, 1:])
    assert pred_seq.shape == (B, S - 1, C, H, W)
    assert torch.equal(pred_seq, seq[:, :-1] + 1.0)
    assert per_loss.shape == (B,)
    assert torch.allclose(per_loss, torch.zeros(B))


@pytest.mark.parametrize("rank", [0, 1])
def test_eval_DDP_loderunner_datastep(
    rank: int, device: torch.device, loss_fn: nn.Module
) -> None:
    """Test eval_DDP step returns all_losses only on rank 0."""
    model = DummyModel()
    B, C, H, W = 2, 1, 2, 2
    start = torch.zeros((B, C, H, W))
    end = torch.zeros((B, C, H, W))
    Dt = torch.ones((B, 1))
    world_size = 4
    end_img, pred_img, all_losses = eval_DDP_loderunner_datastep(
        (start, end, Dt), model, loss_fn, device, rank, world_size
    )
    assert torch.equal(end_img, end)
    assert torch.equal(pred_img, start + 1.0)
    if rank == 0:
        assert all_losses.shape == (world_size * B,)
    else:
        assert all_losses is None


def test_eval_loderunner_datastep_cylex(
    device: torch.device, loss_fn: nn.Module
) -> None:
    """Cylex eval datastep uses channel_map from the 5-tuple data."""
    model = DummyModel()
    B, C, H, W = 2, 1, 2, 2
    start = torch.zeros((B, C, H, W))
    end = torch.zeros((B, C, H, W))
    Dt = torch.ones((B, 1))
    channel_map = [0]

    end_img, pred_img, per_loss = eval_loderunner_datastep_cylex(
        (start, channel_map, end, channel_map, Dt),
        model,
        loss_fn,
        device,
        channel_map=None,
    )
    assert torch.equal(end_img, end)
    assert torch.equal(pred_img, start + 1.0)
    assert per_loss.shape == (B,)
    assert torch.allclose(per_loss, torch.ones(B))


@pytest.mark.parametrize("rank", [0, 1])
def test_train_DDP_loderunner_datastep_cylex(
    rank: int,
    device: torch.device,
    loss_fn: nn.Module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cylex DDP train returns concatenated losses only on rank 0."""
    # Avoid CUDA calls on CPU runners.
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    model = DummyModel()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    B, C, H, W = 2, 1, 2, 2
    start = torch.zeros((B, C, H, W))
    end = torch.zeros((B, C, H, W))
    Dt = torch.ones((B, 1))
    channel_map = [0, 1]

    world_size = 3
    end_img, pred_img, all_losses = train_DDP_loderunner_datastep_cylex(
        (start, channel_map, end, channel_map, Dt),
        model,
        optimizer,
        loss_fn,
        device,
        rank,
        world_size,
    )

    assert torch.equal(end_img, end)
    assert torch.equal(pred_img, start + 1.0)

    if rank == 0:
        assert all_losses is not None
        assert all_losses.shape == (world_size * B,)
    else:
        assert all_losses is None


@pytest.mark.parametrize("rank", [0, 1])
def test_eval_DDP_loderunner_datastep_cylex(
    rank: int,
    device: torch.device,
    loss_fn: nn.Module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cylex DDP eval returns concatenated losses only on rank 0."""
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    model = DummyModel()
    B, C, H, W = 2, 1, 2, 2
    start = torch.zeros((B, C, H, W))
    end = torch.zeros((B, C, H, W))
    Dt = torch.ones((B, 1))
    channel_map = [0, 1]

    world_size = 4
    end_img, pred_img, all_losses = eval_DDP_loderunner_datastep_cylex(
        (start, channel_map, end, channel_map, Dt),
        model,
        loss_fn,
        device,
        rank,
        world_size,
    )

    assert torch.equal(end_img, end)
    assert torch.equal(pred_img, start + 1.0)

    if rank == 0:
        assert all_losses is not None
        assert all_losses.shape == (world_size * B,)
    else:
        assert all_losses is None
