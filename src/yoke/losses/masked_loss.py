"""Modules that define masked losses."""

from collections.abc import Iterable
from typing import Callable

import torch
import torchvision.transforms.functional as tforms


class CroppedLoss2D(torch.nn.Module):
    """Wrapper to crop inputs before computing loss.

    Args:
        loss_fxn (Callable): Function that accepts positional args corresponding
            to a prediction and a target value.
        crop (Iterable): (top, left, height, width) crop passed to
            torchvision.transforms.functional.crop()
    """

    def __init__(
        self,
        loss_fxn: Callable,
        crop: Iterable,
    ) -> None:
        """Initialize cropped loss."""
        super().__init__()
        self.loss_fxn = loss_fxn
        self.crop = crop

    def forward(self, input: torch.tensor, target: torch.tensor) -> torch.tensor:
        """Compute the masked loss."""
        return self.loss_fxn(
            tforms.crop(input, *self.crop), tforms.crop(target, *self.crop)
        )


class MaskedLossMultiplicative(torch.nn.Module):
    """Wrapper to mask loss function by a multiplicative mask on its inputs.

    Args:
        loss_fxn (Callable): Function that accepts positional args corresponding
            to a prediction and a target value.
        mask (torch.tensor): Mask that loss_fxn inputs are multplied by before
            computing loss.
    """

    def __init__(
        self, loss_fxn: Callable, mask: torch.tensor = torch.tensor(1.0)
    ) -> None:
        """Initialize masked loss."""
        super().__init__()
        self.loss_fxn = loss_fxn
        self.register_buffer("mask", mask)

    def forward(self, input: torch.tensor, target: torch.tensor) -> torch.tensor:
        """Compute the masked loss."""
        return self.loss_fxn(input * self.mask, target * self.mask)
