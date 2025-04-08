"""This loss function is a per-channel normalized version of mean squared error."""

import torch
import torch.nn as nn


class NormalizedMSELoss(nn.Module):
    """Per-channel normalized mean squared error loss.

    This loss function normalizes the input and target tensors per channel
    before computing the mean squared error. The normalization is done by
    subtracting the mean and dividing by the standard deviation of the target
    tensor, with a small epsilon added to the standard deviation to avoid
    division by zero.

    Args:
        eps (float): A small value to avoid division by zero.
        reduction (str): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. Default: 'none'.
    """

    def __init__(self, eps: float = 1e-8, reduction: str = "none") -> None:
        """Initialize the NormalizedMSELoss.

        Args:
            eps (float): A small value to avoid division by zero.
            reduction (str): Specifies the reduction to apply to the output:
                'none' | 'mean' | 'sum'. Default: 'none'.
        """
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the normalized mean squared error loss.

        Args:
            pred (torch.Tensor): The predicted tensor.
            target (torch.Tensor): The target tensor.

        Returns:
            torch.Tensor: The computed loss.
        """
        target_mean = target.mean(dim=(0, 2, 3), keepdim=True)
        target_std = target.std(dim=(0, 2, 3), keepdim=True) + self.eps

        pred_norm = (pred - target_mean) / target_std
        target_norm = (target - target_mean) / target_std

        loss = (pred_norm - target_norm) ** 2
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
