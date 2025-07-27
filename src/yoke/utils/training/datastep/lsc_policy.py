"""Single-step training and evaluation functions for lsc240420 policy networks."""

import torch
import torch.nn as nn
import torch.distributed as dist


def train_lsc_policy_datastep(
    data: tuple,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    rank: int,
    world_size: int,
    blocks,  # Temporary list of unfrozen blocks in network for grad observation.
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A DDP-compatible training step for LSC Gaussian policy.

    Args:
        data (tuple): tuple of model input and corresponding ground truth
        model (loaded pytorch model): model to train
        optimizer (torch.optim): optimizer for training set
        loss_fn (torch.nn Loss Function): loss function for training set
        device (torch.device): device index to select
        rank (int): Rank of device
        world_size (int): Number of total DDP processes

    Returns:
        x_true (torch.Tensor): Ground truth end vector
        pred_mean (torch.Tensor): Predicted end vector mean
        all_losses (torch.Tensor): Concatenated per-sample losses from all processes

    """
    # Set model to train mode
    model.train()

    # Extract data
    state_y, stateH, targetH, x_true = data
    state_y = state_y.to(device, non_blocking=True)
    stateH = stateH.to(device, non_blocking=True)
    targetH = targetH.to(device, non_blocking=True)
    x_true = x_true.to(device, non_blocking=True)

    # Forward pass
    pred_distribution = model(state_y, stateH, targetH)
    pred_mean = pred_distribution.mean

    # Compute loss
    loss = loss_fn(pred_mean, x_true)
    per_sample_loss = loss.mean(dim=1)  # Per-sample loss

    # Backward pass and optimization
    optimizer.zero_grad(set_to_none=True)
    loss.mean().backward()

    if rank == 0:
        # Report gradient norms on unfrozen blocks
        for blk_name, blk_match in blocks:
            g2 = 0.0
            for n, p in model.module.named_parameters():
                if p.requires_grad and blk_match(n) and p.grad is not None:
                    #print(f'{n} has grad!')
                    g2 += p.grad.norm().item()
            print(f'{blk_name:12s} grad-norm: {g2:.10f}')

    optimizer.step()

    # Gather per-sample losses from all processes
    gathered_losses = [torch.zeros_like(per_sample_loss) for _ in range(world_size)]
    dist.all_gather(gathered_losses, per_sample_loss)

    # Rank 0 concatenates and saves or returns all losses
    if rank == 0:
        all_losses = torch.cat(gathered_losses, dim=0)  # Shape: (total_batch_size,)
    else:
        all_losses = None

    return x_true, pred_mean, all_losses


def eval_lsc_policy_datastep(
    data: tuple,
    model: nn.Module,
    loss_fn: nn.Module,
    device: torch.device,
    rank: int,
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A DDP-compatible evaluation step.

    Args:
        data (tuple): tuple of model input and corresponding ground truth
        model (loaded pytorch model): model to train
        loss_fn (torch.nn Loss Function): loss function for training set
        device (torch.device): device index to select
        rank (int): Rank of device
        world_size (int): Total number of DDP processes

    Returns:
        x_true (torch.Tensor): Ground truth end vector
        pred_mean (torch.Tensor): Predicted end vector mean
        all_losses (torch.Tensor): Concatenated per-sample losses from all processes
    """
    # Set model to evaluation mode
    model.eval()

    # Extract data
    state_y, stateH, targetH, x_true = data
    state_y = state_y.to(device, non_blocking=True)
    stateH = stateH.to(device, non_blocking=True)
    targetH = targetH.to(device, non_blocking=True)
    x_true = x_true.to(device, non_blocking=True)

    # Forward pass
    with torch.no_grad():
        pred_distribution = model(state_y, stateH, targetH)

    pred_mean = pred_distribution.mean

    # Compute loss
    loss = loss_fn(pred_mean, x_true)
    per_sample_loss = loss.mean(dim=1)  # Per-sample loss

    # Gather per-sample losses from all processes
    gathered_losses = [torch.zeros_like(per_sample_loss) for _ in range(world_size)]
    dist.all_gather(gathered_losses, per_sample_loss)

    # Rank 0 concatenates and saves or returns all losses
    if rank == 0:
        all_losses = torch.cat(gathered_losses, dim=0)  # Shape: (total_batch_size,)
    else:
        all_losses = None

    return x_true, pred_mean, all_losses
