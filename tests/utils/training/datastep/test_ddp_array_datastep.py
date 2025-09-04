"""DDP datastep tests for array-output models.

This suite mirrors the style of `test_array_datastep.py` and focuses on the
distributed variants: `train_DDP_array_datastep` and `eval_DDP_array_datastep`.
"""

from __future__ import annotations

import copy
from typing import Callable

import pytest
import torch
from torch import nn, optim

from yoke.utils.training.datastep.array_output import (
        train_DDP_array_datastep,
        eval_DDP_array_datastep,
    )


# Type alias for the all_gather stub signature.
AllGatherFn = Callable[[list[torch.Tensor], torch.Tensor], None]


def _fake_all_gather() -> AllGatherFn:
    """Return a minimal stand-in for torch.distributed.all_gather.

    The stub copies the input tensor into each preallocated output tensor entry
    without requiring initialization of a process group.

    Returns:
        A callable matching the distributed `all_gather` signature used here.
    """
    def _stub(out_tensors: list[torch.Tensor], in_tensor: torch.Tensor) -> None:
        for i in range(len(out_tensors)):
            out_tensors[i].copy_(in_tensor)
    return _stub


class TestDDPArrayDataStep:
    """Test cases for train/eval DDP datasteps on array-output models."""

    def test_train_ddp_array_datastep_basic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """DDP train returns truth, pred, and aggregated per-sample losses.

        Uses a world size of 1 with a stubbed `all_gather`, so the aggregated
        losses match the local per-sample losses.
        """
        monkeypatch.setattr(
            "torch.distributed.all_gather", _fake_all_gather(), raising=True
        )

        device = torch.device("cpu")
        batch_size, c, h, w = 2, 1, 2, 2
        inpt = torch.ones(batch_size, c, h, w)
        truth = torch.zeros(batch_size, c, h, w)

        model = nn.Conv2d(c, c, kernel_size=1, bias=False)
        # Make deterministic: weight=eye -> prediction equals input.
        with torch.no_grad():
            model.weight.fill_(1.0)

        optimizer = optim.SGD(model.parameters(), lr=0.1)
        loss_fn = nn.MSELoss(reduction="none")

        truth_out, pred_out, all_losses = train_DDP_array_datastep(
            (inpt, truth), model, optimizer, loss_fn, device, rank=0, world_size=1
        )

        # Shapes and identity of tensors.
        assert truth_out.shape == truth.shape
        assert pred_out.shape == inpt.shape
        # Per-sample L2: each sample is all ones vs zeros -> loss=1 per element.
        # Mean over spatial dims -> ones per sample.
        assert all_losses.shape == (batch_size,)
        assert torch.allclose(all_losses, torch.ones(batch_size))

    def test_train_ddp_updates_parameters(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """DDP training should update model parameters."""
        monkeypatch.setattr(
            "torch.distributed.all_gather", _fake_all_gather(), raising=True
        )
        device = torch.device("cpu")
        batch_size, c, h, w = 2, 1, 4, 4
        inpt = torch.randn(batch_size, c, h, w)
        truth = torch.zeros_like(inpt)
        model = nn.Conv2d(c, c, kernel_size=1, bias=False)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss(reduction="none")

        initial = copy.deepcopy(list(model.parameters()))
        train_DDP_array_datastep(
            (inpt, truth), model, optimizer, loss_fn, device, rank=0, world_size=1
        )
        after = list(model.parameters())
        diffs = [not torch.allclose(a, b) for a, b in zip(initial, after)]
        assert any(diffs)

    def test_train_ddp_scalar_loss_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reduction to a scalar loss should raise ValueError in DDP train."""
        monkeypatch.setattr(
            "torch.distributed.all_gather", _fake_all_gather(), raising=True
        )
        device = torch.device("cpu")
        batch_size, c, h, w = 2, 1, 2, 2
        inpt = torch.zeros(batch_size, c, h, w)
        truth = torch.ones_like(inpt)
        model = nn.Conv2d(c, c, kernel_size=1, bias=False)
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        loss_fn = nn.MSELoss(reduction="mean")

        with pytest.raises(ValueError):
            _ = train_DDP_array_datastep(
                (inpt, truth),
                model,
                optimizer,
                loss_fn,
                device,
                rank=0,
                world_size=1,
            )

    def test_eval_ddp_array_datastep_basic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """DDP eval returns truth, pred, and aggregated per-sample losses.

        Checks that calling eval does not modify parameters and the aggregated
        losses equal local per-sample losses for world size 1.
        """
        monkeypatch.setattr(
            "torch.distributed.all_gather", _fake_all_gather(), raising=True
        )
        device = torch.device("cpu")
        batch_size, c, h, w = 3, 2, 1, 1
        inpt = torch.arange(batch_size * c * h * w, dtype=torch.float32).reshape(
            batch_size, c, h, w
        )
        truth = inpt.clone()
        model = nn.Identity()  # No parameters
        loss_fn = nn.MSELoss(reduction="none")

        truth_out, pred_out, all_losses = eval_DDP_array_datastep(
            (inpt, truth), model, None, loss_fn, device, rank=0, world_size=1
        )

        assert torch.allclose(truth_out, truth)
        assert torch.allclose(pred_out, inpt)
        assert torch.allclose(all_losses, torch.zeros(batch_size))

    def test_eval_ddp_does_not_update_parameters(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """DDP eval should not change model parameters."""
        monkeypatch.setattr(
            "torch.distributed.all_gather", _fake_all_gather(), raising=True
        )
        device = torch.device("cpu")
        batch_size, c, h, w = 2, 1, 4, 4
        inpt = torch.randn(batch_size, c, h, w)
        truth = torch.randn(batch_size, c, h, w)
        model = nn.Conv2d(c, c, kernel_size=1, bias=False)
        loss_fn = nn.MSELoss(reduction="none")

        initial = copy.deepcopy(list(model.parameters()))
        _ = eval_DDP_array_datastep(
            (inpt, truth), model, None, loss_fn, device, rank=0, world_size=1
        )
        after = list(model.parameters())
        for init, aft in zip(initial, after):
            assert torch.allclose(init, aft)
