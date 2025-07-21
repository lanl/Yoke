"""Utilities for managing model parameters, their state, and configurations."""

from torch import nn


def count_torch_params(model: nn.Module, trainable: bool = True) -> int:
    """Count parameters in a pytorch model.

    Args:
        model (nn.Module): Model to count parameters for.
        trainable (bool): If TRUE, count only trainable parameters.

    """
    plist = []
    for p in model.parameters():
        if trainable:
            if p.requires_grad:
                plist.append(p.numel())
            else:
                pass
        else:
            plist.append(p.numel())

    return sum(plist)


def freeze_torch_params(model: nn.Module) -> None:
    """Freeze all parameters in a PyTorch model in place.

    Args:
        model (nn.Module): model to freeze.

    """
    for p in model.parameters():
        if hasattr(p, "requires_grad"):
            p.requires_grad = False
