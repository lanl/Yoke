"""Tests for noise schedulers used in diffusion models.

This module contains comprehensive unit tests for the VPCosineNoiseSchedule
class, which implements a variance-preserving (VP) cosine noise schedule for
diffusion models. Tests cover:

- Alpha and sigma coefficient computation
- Variance-preserving property (alpha^2 + sigma^2 = 1)
- Forward diffusion process (adding noise)
- Noise removal (denoising)
- Edge cases at tau=0 (no noise) and tau=1 (pure noise)
- Pre-sampled noise handling
"""

import pytest
import torch

from yoke.utils.diffusion.noise_schedulers import VPCosineNoiseSchedule


@pytest.fixture
def noise_schedule() -> VPCosineNoiseSchedule:
    """Create VPCosineNoiseSchedule fixture."""
    return VPCosineNoiseSchedule()


def test_noise_schedule_alpha(noise_schedule: VPCosineNoiseSchedule) -> None:
    """Test noise schedule alpha coefficient computation."""
    tau = torch.tensor([0.0, 0.5, 1.0])
    alpha = noise_schedule.alpha(tau)

    assert alpha.shape == tau.shape
    assert torch.allclose(alpha[0], torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(alpha[-1], torch.tensor(0.0), atol=1e-6)


def test_noise_schedule_sigma(noise_schedule: VPCosineNoiseSchedule) -> None:
    """Test noise schedule sigma coefficient computation."""
    tau = torch.tensor([0.0, 0.5, 1.0])
    sigma = noise_schedule.sigma(tau)

    assert sigma.shape == tau.shape
    assert torch.allclose(sigma[0], torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(sigma[-1], torch.tensor(1.0), atol=1e-6)


def test_noise_schedule_variance_preserving(
    noise_schedule: VPCosineNoiseSchedule,
) -> None:
    """Test variance-preserving property: alpha^2 + sigma^2 = 1."""
    tau = torch.linspace(0.0, 1.0, 100)
    alpha = noise_schedule.alpha(tau)
    sigma = noise_schedule.sigma(tau)

    variance_sum = alpha**2 + sigma**2

    assert torch.allclose(variance_sum, torch.ones_like(variance_sum))


def test_forward_diffusion(noise_schedule: VPCosineNoiseSchedule) -> None:
    """Test forward diffusion process."""
    batch_size = 2
    y = torch.randn(batch_size, 3, 64, 64)
    tau = torch.rand(batch_size)

    y_tau, noise = noise_schedule.forward_diffusion(y, tau)

    assert y_tau.shape == y.shape
    assert noise.shape == y.shape
    assert not torch.isnan(y_tau).any()
    assert not torch.isnan(noise).any()


def test_forward_diffusion_with_presampled_noise(
    noise_schedule: VPCosineNoiseSchedule,
) -> None:
    """Test forward diffusion with pre-sampled noise."""
    batch_size = 2
    y = torch.randn(batch_size, 3, 64, 64)
    tau = torch.rand(batch_size)
    noise_input = torch.randn_like(y)

    y_tau, noise_output = noise_schedule.forward_diffusion(y, tau, noise=noise_input)

    assert torch.allclose(noise_input, noise_output)


def test_remove_noise(noise_schedule: VPCosineNoiseSchedule) -> None:
    """Test noise removal (denoising)."""
    batch_size = 2
    y = torch.randn(batch_size, 3, 64, 64)
    tau = torch.rand(batch_size)

    # Add noise
    y_tau, noise = noise_schedule.forward_diffusion(y, tau)

    # Remove noise
    y_pred = noise_schedule.remove_noise(y_tau, tau, noise)

    # Should approximately recover original (with some numerical error)
    assert y_pred.shape == y.shape
    assert not torch.isnan(y_pred).any()
    # Check that denoising brings us closer to original
    assert torch.allclose(y_pred, y, atol=1e-4)


def test_diffusion_at_tau_zero(
    noise_schedule: VPCosineNoiseSchedule,
) -> None:
    """Test diffusion at tau=0 (no noise)."""
    y = torch.randn(2, 3, 64, 64)
    tau = torch.zeros(2)

    y_tau, noise = noise_schedule.forward_diffusion(y, tau)

    # At tau=0, should have alpha=1, sigma=0, so y_tau ≈ y
    assert torch.allclose(y_tau, y, atol=1e-5)


def test_diffusion_at_tau_one(
    noise_schedule: VPCosineNoiseSchedule,
) -> None:
    """Test diffusion at tau=1 (pure noise)."""
    y = torch.randn(2, 3, 64, 64)
    tau = torch.ones(2)

    y_tau, noise = noise_schedule.forward_diffusion(y, tau)

    # At tau=1, should have alpha=0, sigma=1, so y_tau ≈ noise
    assert torch.allclose(y_tau, noise, atol=1e-5)
