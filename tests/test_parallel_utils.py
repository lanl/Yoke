"""Tests for the Parallel Utilities module."""

import pytest
import torch
import torch.nn as nn
from torch.testing import assert_close
from yoke.utils.parallel import LodeRunner_DataParallel


# Dummy model for testing
class DummyModel(nn.Module):
    """Dummy model to test parallel wrapper with."""

    def forward(
        self,
        start_img: torch.Tensor,
        in_vars: torch.Tensor,
        out_vars: torch.Tensor,
        Dt: torch.Tensor,
    ) -> torch.Tensor:
        """Dummy forward model evaluation."""
        # Example operation: apply per-channel addition and scale by Dt
        batch_size, channels, _, _ = start_img.shape

        # Broadcast in_vars to match [B, C, H, W]
        in_vars = in_vars.view(1, channels, 1, 1)

        # Broadcast out_vars to match [B, C, H, W]
        out_vars = out_vars.view(1, channels, 1, 1)

        # Broadcast Dt to match [B, C, H, W]
        Dt = Dt.view(batch_size, 1, 1, 1)

        return start_img + in_vars + out_vars * Dt


@pytest.fixture
def setup_environment() -> LodeRunner_DataParallel:
    """Fixture to create a dummy LodeRunner_DataParallel instance."""
    model = DummyModel()
    return LodeRunner_DataParallel(model)


def test_no_device_ids(setup_environment: LodeRunner_DataParallel) -> None:
    """Test forward pass when device_ids is empty."""
    # Mock data
    batch_size, channels, height, width = 4, 3, 32, 32
    start_img = torch.randn(batch_size, channels, height, width)
    in_vars = torch.randn(channels)
    out_vars = torch.randn(channels)
    Dt = torch.randn(batch_size, 1)

    # Set device_ids to None
    setup_environment.device_ids = []
    output = setup_environment(start_img, in_vars, out_vars, Dt)

    # Expected output
    in_vars_broadcast = in_vars.view(1, channels, 1, 1)
    out_vars_broadcast = out_vars.view(1, channels, 1, 1)
    Dt_broadcast = Dt.view(batch_size, 1, 1, 1)
    expected_output = start_img + in_vars_broadcast + out_vars_broadcast * Dt_broadcast

    assert_close(output, expected_output)


def test_forward_invalid_input(setup_environment: LodeRunner_DataParallel) -> None:
    """Test forward pass with invalid inputs."""
    # Pass fewer inputs than required
    with pytest.raises(IndexError):
        setup_environment(torch.randn(4, 3, 32, 32))


def test_forward_mismatched_shapes(setup_environment: LodeRunner_DataParallel) -> None:
    """Test forward pass with mismatched input shapes."""
    # Mock data with mismatched batch dimensions
    start_img = torch.randn(4, 3, 32, 32)
    in_vars = torch.randn(3)
    out_vars = torch.randn(3)
    Dt = torch.randn(8, 1)  # Batch size does not match

    with pytest.raises(RuntimeError):
        setup_environment(start_img, in_vars, out_vars, Dt)
