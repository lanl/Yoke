"""Tests for scalar-output datastep."""

import torch

from yoke.utils.training.datastep.scalar_output import (
    train_scalar_datastep,
    eval_scalar_datastep,
)


class DummyModel(torch.nn.Module):
    """Simple linear model mapping an input vector to a scalar output."""

    def __init__(self, input_dim: int) -> None:
        """Initialize the linear layer.

        Args:
            input_dim: Number of features in the input tensor.
        """
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Tensor of shape (batch_size, 1) with scalar predictions.
        """
        return self.linear(x)


def get_dummy_data(batch_size: int, input_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate random input and ground truth tensors.

    Args:
        batch_size: Number of samples in the batch.
        input_dim: Number of features per sample.

    Returns:
        Tuple of:
            x: Float tensor of shape (batch_size, input_dim).
            y: Float tensor of shape (batch_size,) representing ground truth.
    """
    x = torch.randn(batch_size, input_dim)
    y = torch.randn(batch_size)
    return x, y


def test_train_scalar_datastep_shapes_and_training_mode() -> None:
    """Test train_scalar_datastep returns correct shapes and updates parameters.

    Verifies that:
      - Model is left in train mode.
      - Returned truth, pred, loss all have shape (batch_size, 1).
      - truth matches original y unsqueezed.
      - loss equals loss_fn(pred, truth).
      - Model parameters change after optimizer.step().
    """
    device = torch.device("cpu")
    batch_size, input_dim = 4, 3
    model = DummyModel(input_dim)
    loss_fn = torch.nn.MSELoss(reduction="none")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    data = get_dummy_data(batch_size, input_dim)

    # Snapshot initial weights
    initial_weight = model.linear.weight.clone().detach()

    truth, pred, loss = train_scalar_datastep(data, model, optimizer, loss_fn, device)

    # Model should be in training mode
    assert model.training, "Expected model.training == True after train step"

    # All outputs should have shape (batch_size, 1)
    assert truth.shape == (batch_size, 1)
    assert pred.shape == (batch_size, 1)
    assert loss.shape == (batch_size, 1)

    # Ground truth was unsqueezed and cast to float32
    original_y = data[1].to(torch.float32).unsqueeze(-1)
    assert torch.allclose(truth.cpu(), original_y, atol=1e-6)

    # Loss should match direct application of loss_fn
    expected_loss = loss_fn(pred, truth)
    assert torch.allclose(loss, expected_loss)

    # Confirm that model weights were updated by optimizer.step()
    updated_weight = model.linear.weight.clone().detach()
    assert not torch.allclose(initial_weight, updated_weight), (
        "Weights should change after training step"
    )


def test_eval_scalar_datastep_shapes_and_eval_mode() -> None:
    """Test eval_scalar_datastep returns correct shapes and sets eval mode.

    Verifies that:
      - Model is left in evaluation mode.
      - Returned truth, pred, loss all have shape (batch_size, 1).
      - truth matches original y unsqueezed.
      - loss equals loss_fn(pred, truth).
    """
    device = torch.device("cpu")
    batch_size, input_dim = 5, 2
    model = DummyModel(input_dim)
    loss_fn = torch.nn.MSELoss(reduction="none")

    # Force model into train mode before eval call
    model.train()
    data = get_dummy_data(batch_size, input_dim)

    truth, pred, loss = eval_scalar_datastep(data, model, loss_fn, device)

    # Model should be in eval mode
    assert not model.training, "Expected model.training == False after eval step"

    # All outputs should have shape (batch_size, 1)
    assert truth.shape == (batch_size, 1)
    assert pred.shape == (batch_size, 1)
    assert loss.shape == (batch_size, 1)

    # Ground truth was unsqueezed and cast to float32
    original_y = data[1].to(torch.float32).unsqueeze(-1)
    assert torch.allclose(truth.cpu(), original_y, atol=1e-6)

    # Loss should match direct application of loss_fn
    expected_loss = loss_fn(pred, truth)
    assert torch.allclose(loss, expected_loss)
