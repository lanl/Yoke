"""Functions to perform a training or evaluation step on a model.

This module is specific to networks with single, multi-dimensional array outputs.
"""

import torch
from torch import nn
import torch.distributed as dist


def train_array_datastep(
    data: tuple,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Training step for network whose output is a multi-dimensional array.

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
    # NOTE: Loss should expect shape: (B, C0, C1, ..., CN) with reduction='none' so
    # per-sample loss can be returned
    loss = loss_fn(pred, truth)

    if loss.ndim == 0:
        # loss_fn was reduced to a scalar; can't compute per-sample loss
        raise ValueError(
            'loss_fn returned a scalar. Set reduction=none (or equivalent) to compute'
            'per-sample losses.'
        )

    reduce_dims = tuple(range(1, loss.ndim))  # (1, 2, ..., N)
    per_sample_loss = loss.mean(dim=reduce_dims)  # Shape: (batch_size,)

    # Perform backpropagation and update the weights
    # optimizer.zero_grad()
    optimizer.zero_grad(set_to_none=True)  # Possible speed-up
    loss.mean().backward()
    optimizer.step()

    return truth, pred, per_sample_loss


def train_DDP_array_datastep(
    data: tuple,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    rank: int,
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A DDP-compatible training step for array output models.

    Args:
        data (tuple): tuple of model input and corresponding ground truth
        model (loaded pytorch model): model to train
        optimizer (torch.optim): optimizer for training set
        loss_fn (torch.nn Loss Function): loss function for training set
        device (torch.device): device index to select
        rank (int): Rank of device
        world_size (int): Number of total DDP processes

    Returns:
        truth (torch.Tensor): Ground truth end vector
        pred (torch.Tensor): Predicted end vector
        all_losses (torch.Tensor): Concatenated per-sample losses from all processes

    """
    # Set model to train mode
    model.train()

    # Extract data
    (inpt, truth) = data
    inpt = inpt.to(device, non_blocking=True)
    truth = truth.to(device, non_blocking=True)

    # Forward pass
    pred = model(inpt)
    # NOTE: Loss should expect shape: (B, C0, C1, ..., CN) with reduction='none' so
    # per-sample loss can be returned
    loss = loss_fn(pred, truth)

    if loss.ndim == 0:
        # loss_fn was reduced to a scalar; can't compute per-sample loss
        raise ValueError(
            'loss_fn returned a scalar. Set reduction=none (or equivalent) to compute'
            'per-sample losses.'
        )

    reduce_dims = tuple(range(1, loss.ndim))  # (1, 2, ..., N)
    per_sample_loss = loss.mean(dim=reduce_dims)  # Shape: (batch_size,)

    # Backward pass and optimization
    optimizer.zero_grad(set_to_none=True)
    loss.mean().backward()

    # Step the optimizer
    optimizer.step()

    # Gather per-sample losses from all processes
    gathered_losses = [torch.zeros_like(per_sample_loss) for _ in range(world_size)]
    dist.all_gather(gathered_losses, per_sample_loss)

    # Rank 0 concatenates and saves or returns all losses
    if rank == 0:
        all_losses = torch.cat(gathered_losses, dim=0)  # Shape: (total_batch_size,)
    else:
        all_losses = None

    return truth, pred, all_losses


def eval_array_datastep(
    data: tuple, model: nn.Module, loss_fn: nn.Module, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluation on a single batch of a network whose output is an array.

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
    reduce_dims = tuple(range(1, loss.ndim))  # (1, 2, ..., N)
    per_sample_loss = loss.mean(dim=reduce_dims)  # Shape: (batch_size,)

    return truth, pred, per_sample_loss


def eval_DDP_array_datastep(
    data: tuple,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    rank: int,
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A DDP-compatible evaluation step for array output models.

    Args:
        data (tuple): tuple of model input and corresponding ground truth
        model (loaded pytorch model): model to train
        optimizer (torch.optim): optimizer for training set
        loss_fn (torch.nn Loss Function): loss function for training set
        device (torch.device): device index to select
        rank (int): Rank of device
        world_size (int): Number of total DDP processes

    Returns:
        truth (torch.Tensor): Ground truth end vector
        pred (torch.Tensor): Predicted end vector
        all_losses (torch.Tensor): Concatenated per-sample losses from all processes

    """
    # Set model to eval mode
    model.eval()

    # Extract data
    (inpt, truth) = data
    inpt = inpt.to(device, non_blocking=True)
    truth = truth.to(device, non_blocking=True)

    # Forward pass
    with torch.no_grad():
        pred = model(inpt)

    # NOTE: Loss should expect shape: (B, C0, C1, ..., CN) with reduction='none' so
    # per-sample loss can be returned
    loss = loss_fn(pred, truth)

    reduce_dims = tuple(range(1, loss.ndim))  # (1, 2, ..., N)
    per_sample_loss = loss.mean(dim=reduce_dims)  # Shape: (batch_size,)

    # Gather per-sample losses from all processes
    gathered_losses = [torch.zeros_like(per_sample_loss) for _ in range(world_size)]
    dist.all_gather(gathered_losses, per_sample_loss)

    # Rank 0 concatenates and saves or returns all losses
    if rank == 0:
        all_losses = torch.cat(gathered_losses, dim=0)  # Shape: (total_batch_size,)
    else:
        all_losses = None

    return truth, pred, all_losses
