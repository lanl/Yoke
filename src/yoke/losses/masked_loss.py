"""Modules that define masked losses, e.g., losses that are only computed in part of an image"""

from typing import Callable

import torch


class MaskedLossMultiplicative(torch.nn.Module):
    """
    Wrapper to mask loss function by a multiplicative mask on its inputs.

    Args:
        loss_fxn (Callable): Function that accepts positional args corresponding
            to a prediction and a target value.
        mask (torch.tensor): Mask that loss_fxn inputs are multplied by before
            computing loss.
    """

    def __init__(
        self, loss_fxn: Callable, mask: torch.tensor = torch.tensor(1.0)
    ) -> None:
        super().__init__()
        self.loss_fxn = loss_fxn
        self.register_buffer("mask", mask)

    def forward(self, input: torch.tensor, target: torch.tensor):
        return self.loss_fxn(input * self.mask, target * self.mask)
