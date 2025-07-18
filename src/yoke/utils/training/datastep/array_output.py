"""Functions to perform a training or evaluation step on a model.

This module is specific to networks with single, multi-dimensional array outputs.
"""

import torch
from torch import nn


def train_array_datastep(
    data: tuple,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Training step for network whose output is a CxHxW array.

    Args:
        data (tuple): tuple of model input and corresponding ground truth
        model (loaded pytorch model): model to train
        optimizer (torch.optim): optimizer for training set
        loss_fn (torch.nn Loss Function): loss function for training set
        device (torch.device): device index to select

    Returns:
        truth (torch.Tensor): ground truth tensor
        pred (torch.Tensor): predicted tensor
        per_sample_loss (torch.Tensor): loss for each sample in the batch

    """
    # Set model to train
    model.train()

    # Extract data
    (inpt, truth) = data
    inpt = inpt.to(device, non_blocking=True)
    truth = truth.to(device, non_blocking=True)

    # Perform a forward pass
    # NOTE: If training on GPU model should have already been moved to GPU
    # prior to initalizing optimizer.
    pred = model(inpt)
    loss = loss_fn(pred, truth)

    per_sample_loss = loss.mean(dim=[1, 2, 3])  # Shape: (batch_size,)

    # Perform backpropagation and update the weights
    # optimizer.zero_grad()
    optimizer.zero_grad(set_to_none=True)  # Possible speed-up
    loss.mean().backward()
    optimizer.step()

    return truth, pred, per_sample_loss


def eval_array_datastep(
    data: tuple, model: nn.Module, loss_fn: nn.Module, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluation on a single batch of a network whose output is a CxHxW array.

    Args:
        data (tuple): tuple of model input and corresponding ground truth
        model (loaded pytorch model): model evaluate
        loss_fn (torch.nn Loss Function): loss function for training set
        device (torch.device): device index to select

    Returns:
        truth (torch.Tensor): ground truth tensor
        pred (torch.Tensor): predicted tensor
        per_sample_loss (torch.Tensor): loss for each sample in the batch

    """
    # Set model to eval
    model.eval()

    # Extract data
    (inpt, truth) = data
    inpt = inpt.to(device, non_blocking=True)
    truth = truth.to(device, non_blocking=True)

    # Perform a forward pass
    pred = model(inpt)
    loss = loss_fn(pred, truth)
    per_sample_loss = loss.mean(dim=[1, 2, 3])  # Shape: (batch_size,)

    return truth, pred, per_sample_loss
