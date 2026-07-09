
"""Noise schedulers for diffusion models.

This module provides noise scheduling strategies diffusion processes.
Noise schedulers define how noise is added during the forward
diffusion process and how it is removed during sampling/inference.

The schedulers implement the forward diffusion process:
    y_tau = alpha(tau) * y + sigma(tau) * epsilon

where:
    - tau (τ) ∈ [0,1] is the diffusion time (0 = clean, 1 = pure noise)
    - alpha(tau) and sigma(tau) are noise schedule coefficients

Broadly, noise schedules can be varience-preserving (VP) or variance-exploding (VE).
The variance-preserving constraint is alpha(tau)^2 + sigma(tau)^2 = 1
"""

import math
import torch

class VPCosineNoiseSchedule:
    """Variance-preserving (VP) cosine noise schedule.

    Implements the VP forward diffusion process:
        y_tau = alpha(tau) * y + sigma(tau) * epsilon
    where alpha(tau)^2 + sigma(tau)^2 = 1

    Uses a cosine schedule for smooth interpolation.
    """

    def __init__(self) -> None:
        """Initialization for VP noise schedule."""
        #I don't know if we need an init
        pass

    def alpha(self, tau: torch.Tensor) -> torch.Tensor:
        """Compute coefficient alpha(tau) = cos(pi*tau/2).

        Args:
            tau: Diffusion time in [0, 1], shape (B,) or (B, 1).

        Returns:
            alpha(tau) values, same shape as tau.
        """
        return torch.cos(math.pi * tau / 2.0)

    def sigma(self, tau: torch.Tensor) -> torch.Tensor:
        """Compute coefficient sigma(tau) = sin(pi*tau/2).

        Args:
            tau: Diffusion time in [0, 1], shape (B,) or (B, 1).

        Returns:
            sigma(tau) values, same shape as tau.
        """
        return torch.sin(math.pi * tau / 2.0)

    def forward_diffusion(
        self, y: torch.Tensor, tau: torch.Tensor, noise: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply forward diffusion process.

        Implements: y_tau = alpha(tau) * y + sigma(tau) * noise.

        Args:
            y: Clean target of shape (B, C, H, W).
            tau: Diffusion time in [0, 1], shape (B,).
            noise: Optional pre-sampled noise. If None, samples from N(0, I).

        Returns:
            y_tau: Noised target of shape (B, C, H, W).
            noise: The noise that was added, shape (B, C, H, W).
        """
        if noise is None:
            noise = torch.randn_like(y)

        # Reshape tau for broadcasting: (B,) -> (B, 1, 1, 1)
        tau_expanded = tau.view(-1, 1, 1, 1)

        # Compute coefficients
        alpha_tau = self.alpha(tau_expanded)
        sigma_tau = self.sigma(tau_expanded)

        # Apply VP forward process: y_tau = α(τ)*y + σ(τ)*ε
        y_tau = alpha_tau * y + sigma_tau * noise

        return y_tau, noise

    def remove_noise(
        self, y_tau: torch.Tensor, tau: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """Removes noise from target data.

        Implements: ŷ_0 = (y_tau - sigma(tau)*noise) / alpha(tau)

        Args:
            y_tau: Noised target of shape (B, C, H, W).
            tau: Diffusion time in [0, 1], shape (B,).
            noise: noise of shape (B, C, H, W).

        Returns:
            Denoised target of shape (B, C, H, W).
        """
        # Reshape tau for broadcasting
        tau_expanded = tau.view(-1, 1, 1, 1)

        alpha_tau = self.alpha(tau_expanded)
        sigma_tau = self.sigma(tau_expanded)

        y0_pred = (y_tau - sigma_tau * noise) / (alpha_tau + 1e-8)

        return y0_pred
