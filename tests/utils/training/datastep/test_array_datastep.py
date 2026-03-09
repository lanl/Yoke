"""Tests for datastep on network with array output."""

import copy

import torch
from torch import nn, optim

from yoke.utils.training.datastep.array_output import (
    train_array_datastep,
    eval_array_datastep,
)


class TestArrayDataStep:
    """Test cases for train_array_datastep and eval_array_datastep."""

    def test_train_array_datastep_basic(self) -> None:
        """Test that train_array_datastep returns truth, pred, and per-sample loss.

        In this case we test tensors with correct shapes and values for a simple identity
        model.
        """
        device = torch.device("cpu")
        batch_size, c, h, w = 2, 1, 2, 2
        inpt = torch.ones(batch_size, c, h, w)
        truth = torch.zeros(batch_size, c, h, w)
        model = nn.Conv2d(c, c, kernel_size=1, bias=False)
        model.weight.data.fill_(1.0)
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        loss_fn = nn.MSELoss(reduction="none")

        truth_out, pred_out, per_sample_loss = train_array_datastep(
            (inpt, truth), model, optimizer, loss_fn, device
        )

        # Check shapes.
        assert truth_out.shape == (batch_size, c, h, w)
        assert pred_out.shape == (batch_size, c, h, w)
        assert per_sample_loss.shape == (batch_size,)

        # Identity model and MSE between ones and zeros yields loss of 1.
        assert torch.allclose(pred_out, inpt.to(device))
        assert torch.allclose(truth_out, truth.to(device))
        assert torch.allclose(per_sample_loss, torch.ones(batch_size))

    def test_eval_array_datastep_basic(self) -> None:
        """Test that eval_array_datastep returns truth, pred, and per-sample loss."""
        device = torch.device("cpu")
        batch_size, c, h, w = 3, 2, 1, 1
        inpt = torch.arange(batch_size * c * h * w, dtype=torch.float32).reshape(
            batch_size, c, h, w
        )
        truth = inpt.clone()
        model = nn.Identity()
        loss_fn = nn.MSELoss(reduction="none")

        truth_out, pred_out, per_sample_loss = eval_array_datastep(
            (inpt, truth), model, loss_fn, device
        )

        # Check shapes.
        assert truth_out.shape == (batch_size, c, h, w)
        assert pred_out.shape == (batch_size, c, h, w)
        assert per_sample_loss.shape == (batch_size,)

        # Identity model with identical truth yields zero loss.
        assert torch.allclose(pred_out, truth_out)
        assert torch.allclose(per_sample_loss, torch.zeros(batch_size))

    def test_per_sample_loss_computation(self) -> None:
        """Test that per_sample_loss is computed as the mean of the loss tensor."""
        device = torch.device("cpu")
        batch_size, c, h, w = 4, 1, 2, 2
        inpt = torch.zeros(batch_size, c, h, w)
        truth = torch.ones(batch_size, c, h, w)
        model = nn.Identity()
        loss_fn = nn.MSELoss(reduction="none")

        _, _, per_sample_loss = eval_array_datastep(
            (inpt, truth), model, loss_fn, device
        )

        expected = torch.ones(batch_size)
        assert torch.allclose(per_sample_loss, expected)

    def test_train_updates_model_parameters(self) -> None:
        """Test that train_array_datastep updates model parameters.

        Specifically for the case when loss is non-zero for a simple convolutional model.
        """
        device = torch.device("cpu")
        batch_size, c, h, w = 2, 1, 3, 3
        inpt = torch.ones(batch_size, c, h, w)
        truth = torch.zeros(batch_size, c, h, w)
        model = nn.Conv2d(c, c, kernel_size=1, bias=True)
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        loss_fn = nn.MSELoss(reduction="none")

        initial_params = copy.deepcopy(list(model.parameters()))

        train_array_datastep((inpt, truth), model, optimizer, loss_fn, device)

        updated_params = list(model.parameters())
        differences = [
            not torch.allclose(init, upd)
            for init, upd in zip(initial_params, updated_params)
        ]
        assert any(differences)

    def test_eval_does_not_update_parameters(self) -> None:
        """Test that eval_array_datastep does not modify model parameters."""
        device = torch.device("cpu")
        batch_size, c, h, w = 2, 1, 4, 4
        inpt = torch.randn(batch_size, c, h, w)
        truth = torch.randn(batch_size, c, h, w)
        model = nn.Conv2d(c, c, kernel_size=1, bias=False)
        loss_fn = nn.MSELoss(reduction="none")

        initial_params = copy.deepcopy(list(model.parameters()))

        eval_array_datastep((inpt, truth), model, loss_fn, device)

        after_params = list(model.parameters())
        for init, after in zip(initial_params, after_params):
            assert torch.allclose(init, after)
