"""Functions to perform a training or evaluation step on a model.

This module is specific to networks with single, scalar outputs.
"""

import torch


def train_scalar_datastep(
    data: tuple,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> tuple:
    """Training step for network whose output is a scalar.

    Args:
        data (tuple): tuple of model input and corresponding ground truth
        model (loaded pytorch model): model to train
        optimizer (torch.optim): optimizer for training set
        loss_fn (torch.nn Loss Function): loss function for training set
        device (torch.device): device index to select

    Returns:
        truth (torch.Tensor): ground truth tensor
        pred (torch.Tensor): predicted tensor
        loss (torch.Tensor): per-sample evaluated loss for the data batch

    """
    # Set model to train
    model.train()

    # Extract data
    (inpt, truth) = data
    inpt = inpt.to(device, non_blocking=True)
    # Unsqueeze is necessary for scalar ground-truth output
    truth = truth.to(torch.float32).unsqueeze(-1).to(device, non_blocking=True)

    # Perform a forward pass
    # NOTE: If training on GPU model should have already been moved to GPU
    # prior to initalizing optimizer.
    pred = model(inpt)
    loss = loss_fn(pred, truth)

    # Perform backpropagation and update the weights
    optimizer.zero_grad(set_to_none=True)
    loss.mean().backward()
    optimizer.step()

    return truth, pred, loss


def eval_scalar_datastep(
    data: tuple, model: torch.nn.Module, loss_fn: torch.nn.Module, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluation on a single batch of a network whose output is a scalar.

    Args:
        data (tuple): tuple of model input and corresponding ground truth
        model (loaded pytorch model): model evaluate
        loss_fn (torch.nn Loss Function): loss function for training set
        device (torch.device): device index to select

    Returns:
        truth (torch.Tensor): ground truth tensor
        pred (torch.Tensor): predicted tensor
        loss (torch.Tensor): evaluated loss for the data sample

    """
    # Set model to eval
    model.eval()

    # Extract data
    (inpt, truth) = data
    inpt = inpt.to(device, non_blocking=True)
    truth = truth.to(torch.float32).unsqueeze(-1).to(device, non_blocking=True)

    # Perform a forward pass
    pred = model(inpt)
    loss = loss_fn(pred, truth)

    return truth, pred, loss
