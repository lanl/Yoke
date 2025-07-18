"""Unit tests for the mnist_CNN module in yoke.models.mnist_model.

This module tests the initialization and forward pass of the CNN architecture
for MNIST classification. Tests include shape checks, probability distribution
checks, error handling for invalid inputs, and reproducibility in eval mode.
"""

import pytest
import torch

from yoke.models.mnist_model import mnist_CNN


class TestMnistCNN:
    """Tests for the mnist_CNN neural network."""

    def test_default_initialization(self) -> None:
        """Test that the model initializes with default parameters."""
        model = mnist_CNN()
        # Check that convolutional layers exist
        assert hasattr(model, "conv1")
        assert hasattr(model, "conv2")
        assert hasattr(model, "conv3")
        assert hasattr(model, "conv4")
        # Check that dropout and fully connected layers exist
        assert hasattr(model, "dropout")
        assert hasattr(model, "fc1")
        assert hasattr(model, "fc2")

    def test_forward_shape(self) -> None:
        """Test that forward returns correct shape for default model."""
        model = mnist_CNN()
        batch_size = 4
        input_tensor = torch.randn(batch_size, 1, 28, 28)
        output = model(input_tensor)
        assert isinstance(output, torch.Tensor)
        assert output.shape == (batch_size, 10)

    def test_log_probabilities_sum_to_one(self) -> None:
        """Test output log probabilities exponentiate to probability distribution."""
        model = mnist_CNN()
        input_tensor = torch.randn(3, 1, 28, 28)
        output = model(input_tensor)
        probs = torch.exp(output)
        # Sum of probabilities per sample is close to 1
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)

    def test_custom_conv_sizes(self) -> None:
        """Test that the model works with custom convolution sizes."""
        sizes = (1, 2, 3, 4)
        model = mnist_CNN(*sizes)
        input_tensor = torch.randn(2, 1, 28, 28)
        output = model(input_tensor)
        assert output.shape == (2, 10)

    def test_invalid_input_raises_error(self) -> None:
        """Test that invalid input channel size raises a RuntimeError."""
        model = mnist_CNN()
        # Input with incorrect channel dimension (should be 1)
        bad_input = torch.randn(1, 2, 28, 28)
        with pytest.raises(RuntimeError):
            model(bad_input)

    def test_eval_mode_reproducibility(self) -> None:
        """Test that model in eval mode produces reproducible outputs."""
        model = mnist_CNN()
        model.eval()
        input_tensor = torch.randn(1, 1, 28, 28)
        out1 = model(input_tensor)
        out2 = model(input_tensor)
        assert torch.allclose(out1, out2)
