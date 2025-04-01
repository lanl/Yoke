"""Tests for torch checkpoint functions with dynamic model."""

import os
import tempfile

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from torch import Tensor

# Your checkpoint functions here or imported
from yoke.torch_training_utils import save_model_and_optimizer
from yoke.torch_training_utils import load_model_and_optimizer


class DummyNet(nn.Module):
    """Simple test model."""

    def __init__(self, input_dim: int = 4, output_dim: int = 2) -> None:
        """Initialization."""
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Forward map."""
        return self.linear(x)


@pytest.fixture
def dummy_model_args() -> dict:
    """Dummy model parameters dict."""
    return {"input_dim": 4, "output_dim": 2}


@pytest.fixture
def available_models() -> dict[str, type[nn.Module]]:
    """Dummy available models."""
    return {"DummyNet": DummyNet}


@pytest.fixture
def model_and_optimizer(dummy_model_args: dict) -> tuple[nn.Module, optim.Optimizer]:
    """Model and optimizer initialization function."""
    model = DummyNet(**dummy_model_args)
    optimizer = optim.AdamW(model.parameters(), lr=0.01)
    return model, optimizer


def test_checkpoint_save_and_load(
    model_and_optimizer: tuple[nn.Module, optim.Optimizer],
    dummy_model_args: dict,
    available_models: dict[str, type[nn.Module]],
) -> None:
    """Test saving and reloading, non-DDP."""
    model, optimizer = model_and_optimizer

    # Modify model weights to test restoration
    with torch.no_grad():
        for param in model.parameters():
            param.add_(1.0)

    # Take optimizer step to populate state dict
    dummy_input = torch.randn(1, 4)
    output = model(dummy_input)
    loss = output.sum()
    loss.backward()
    optimizer.step()

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "checkpoint.pth")

        save_model_and_optimizer(
            model=model,
            optimizer=optimizer,
            epoch=5,
            filepath=ckpt_path,
            model_class=DummyNet,
            model_args=dummy_model_args,
        )

        # Create new model/optimizer pair to load into
        new_model = DummyNet(**dummy_model_args)
        new_optimizer = optim.AdamW(new_model.parameters(), lr=0.01)

        loaded_model, loaded_epoch = load_model_and_optimizer(
            filepath=ckpt_path,
            optimizer=new_optimizer,
            available_models=available_models,
            device="cpu",
        )

        # Check epoch was restored
        assert loaded_epoch == 5

        # Compare model parameters
        for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
            assert torch.allclose(p1, p2), "Model parameters not restored correctly"

        # Compare optimizer states
        old_opt_state = optimizer.state_dict()
        new_opt_state = new_optimizer.state_dict()

        for k in old_opt_state["state"].keys():
            for subkey in old_opt_state["state"][k]:
                v1 = old_opt_state["state"][k][subkey]
                v2 = new_opt_state["state"][k][subkey]
                if isinstance(v1, torch.Tensor):
                    assert torch.allclose(v1, v2)
                else:
                    assert v1 == v2
