"""Single-step training and evaluation functions for an lsc240420 reward network."""

import torch
import torch.nn as nn
import torch.distributed as dist


def train_lsc_reward_datastep(
    data: tuple,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    rank: int,
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A DDP-compatible training step for LSC reward network.

    Args:
        data (tuple): tuple of model input and corresponding ground truth
        model (loaded pytorch model): model to train
        optimizer (torch.optim): optimizer for training set
        loss_fn (torch.nn Loss Function): loss function for training set
        device (torch.device): device index to select
        rank (int): Rank of device
        world_size (int): Number of total DDP processes

    Returns:
        reward (torch.Tensor): Ground truth reward
        pred (torch.Tensor): Predicted reward
        all_losses (torch.Tensor): Concatenated per-sample losses from all processes
    """
    # Set model to train mode
    model.train()

    # Extract data
    state_y, stateH, targetH, reward = data
    state_y = state_y.to(device, non_blocking=True)
    stateH = stateH.to(device, non_blocking=True)
    targetH = targetH.to(device, non_blocking=True)
    reward = reward.to(device, non_blocking=True)

    # make reward shape (batch,1) to match pred.shape
    reward = reward.unsqueeze(1)

    # Forward pass
    pred = model(state_y, stateH, targetH)

    # Compute loss
    loss = loss_fn(pred, reward)
    per_sample_loss = loss  # Per-sample loss

    # Backward pass and optimization
    optimizer.zero_grad(set_to_none=True)
    loss.mean().backward()
    optimizer.step()

    # Gather per-sample losses from all processes
    gathered_losses = [torch.zeros_like(per_sample_loss) for _ in range(world_size)]
    dist.all_gather(gathered_losses, per_sample_loss)

    # Rank 0 concatenates and saves or returns all losses
    if rank == 0:
        all_losses = torch.cat(gathered_losses, dim=0)  # Shape: (total_batch_size,)
    else:
        all_losses = None

    return reward, pred, all_losses


def eval_lsc_reward_datastep(
    data: tuple,
    model: nn.Module,
    loss_fn: nn.Module,
    device: torch.device,
    rank: int,
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A DDP-compatible evaluation step for LSC reward network.

    Args:
        data (tuple): tuple of model input and corresponding ground truth
        model (loaded pytorch model): model to train
        loss_fn (torch.nn Loss Function): loss function for training set
        device (torch.device): device index to select
        rank (int): Rank of device
        world_size (int): Total number of DDP processes

    Returns:
        reward (torch.Tensor): Ground truth reward
        pred (torch.Tensor): Predicted reward
        all_losses (torch.Tensor): Concatenated per-sample losses from all processes
    """
    # Set model to evaluation mode
    model.eval()

    # Extract data
    state_y, stateH, targetH, reward = data
    state_y = state_y.to(device, non_blocking=True)
    stateH = stateH.to(device, non_blocking=True)
    targetH = targetH.to(device, non_blocking=True)
    reward = reward.to(device, non_blocking=True)
    # make reward shape (batch,1) to match pred.shape
    reward = reward.unsqueeze(1)

    # Forward pass
    with torch.no_grad():
        pred = model(state_y, stateH, targetH)

    # Compute loss
    loss = loss_fn(pred, reward)
    per_sample_loss = loss  # Per-sample loss

    # Gather per-sample losses from all processes
    gathered_losses = [torch.zeros_like(per_sample_loss) for _ in range(world_size)]
    dist.all_gather(gathered_losses, per_sample_loss)

    # Rank 0 concatenates and saves or returns all losses
    if rank == 0:
        all_losses = torch.cat(gathered_losses, dim=0)  # Shape: (total_batch_size,)
    else:
        all_losses = None

    return reward, pred, all_losses
