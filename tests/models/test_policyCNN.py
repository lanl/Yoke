"""Tests for Gaussian policy with CNN."""

import pytest
import torch
from torch.distributions import MultivariateNormal

from yoke.models.policyCNNmodules import gaussian_policyCNN


@pytest.fixture
def dummy_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Set up dummy inputs for tests."""
    batch_size = 4
    input_vector_size = 28
    img_shape = (1, 1120, 400)
    y = torch.randn(batch_size, input_vector_size)
    h1 = torch.randn(batch_size, *img_shape)
    h2 = torch.randn(batch_size, *img_shape)
    return y, h1, h2


@pytest.fixture
def model() -> gaussian_policyCNN:
    """Fixture to initialize model."""
    return gaussian_policyCNN()


def test_forward_output_type(
    model: gaussian_policyCNN,
    dummy_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """Test forward pass works."""
    y, h1, h2 = dummy_inputs
    output = model(y, h1, h2)
    assert isinstance(output, MultivariateNormal)


def test_output_mean_and_cov_shape(
    model: gaussian_policyCNN,
    dummy_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """Test output mean and covariance shape."""
    y, h1, h2 = dummy_inputs
    output = model(y, h1, h2)
    assert output.mean.shape == (y.size(0), model.output_dim)
    assert output.covariance_matrix.shape == (
        y.size(0),
        model.output_dim,
        model.output_dim,
    )


def test_covariance_is_symmetric(
    model: gaussian_policyCNN,
    dummy_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """Test covariance symmetry."""
    y, h1, h2 = dummy_inputs
    output = model(y, h1, h2)
    cov = output.covariance_matrix
    diff = (cov - cov.transpose(-1, -2)).abs().max()
    assert diff.detach().item() < 1e-5


def test_covariance_positive_definite(
    model: gaussian_policyCNN,
    dummy_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """Test covariance positive definite."""
    y, h1, h2 = dummy_inputs
    output = model(y, h1, h2)
    for mat in output.covariance_matrix:
        eigvals = torch.linalg.eigvalsh(mat)
        assert torch.all(eigvals > 0)


def test_backward_pass(
    model: gaussian_policyCNN,
    dummy_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """Test backpropagation."""
    y, h1, h2 = dummy_inputs
    dist = model(y, h1, h2)
    sample = dist.rsample()
    loss = sample.sum()
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()


def test_different_batch_sizes() -> None:
    """Test batch size handling."""
    for batch_size in [1, 2, 8]:
        model = gaussian_policyCNN()
        y = torch.randn(batch_size, model.input_vector_size)
        h1 = torch.randn(batch_size, *model.img_size)
        h2 = torch.randn(batch_size, *model.img_size)
        dist = model(y, h1, h2)
        assert dist.mean.shape == (batch_size, model.output_dim)
