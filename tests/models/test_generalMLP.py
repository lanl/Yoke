"""Test general MLP layer."""

import pytest
import torch
from torch import nn

from yoke.models.hybridCNNmodules import generalMLP


@pytest.fixture
def default_general_mlp() -> generalMLP:
    """Fixture for default generalMLP instance."""
    return generalMLP()


def test_general_mlp_initialization(default_general_mlp: generalMLP) -> None:
    """Test initialization of the generalMLP."""
    assert isinstance(default_general_mlp, generalMLP)
    assert isinstance(default_general_mlp.LayerList, nn.ModuleList)
    assert len(default_general_mlp.LayerList) > 0


def test_general_mlp_forward_output_shape(default_general_mlp: generalMLP) -> None:
    """Test forward method output shape."""
    batch_size = 8
    input_tensor = torch.rand(batch_size, default_general_mlp.input_dim)
    output = default_general_mlp(input_tensor)
    assert output.shape == (batch_size, default_general_mlp.output_dim)


def test_general_mlp_forward_pass(default_general_mlp: generalMLP) -> None:
    """Test forward pass of generalMLP."""
    batch_size = 8
    input_tensor = torch.rand(batch_size, default_general_mlp.input_dim)
    output = default_general_mlp(input_tensor)
    assert torch.is_tensor(output)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


@pytest.mark.parametrize(
    "input_dim,output_dim,hidden_features",
    [
        (64, 16, (16, 32, 32, 16)),
        (128, 32, (64, 128, 64)),
    ],
)
def test_general_mlp_custom_initialization(
    input_dim: int, output_dim: int, hidden_features: tuple[int, ...]
) -> None:
    """Test generalMLP initialization with custom parameters."""
    model = generalMLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_feature_list=hidden_features,
    )
    assert model.input_dim == input_dim
    assert model.output_dim == output_dim
    assert model.hidden_feature_list == hidden_features
