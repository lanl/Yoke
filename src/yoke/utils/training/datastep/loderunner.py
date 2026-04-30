"""Functions for training and evaluation datasteps for Loderunner.

This module provides functions to perform single training or evaluation steps on a
Loderunner model. The training methods for Loderunner are still developing so multiple
possible training methods are provided.
"""

import torch
import torch.distributed as dist
import random


def train_loderunner_datastep(
    data: tuple,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device,
    channel_map: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A training step for which the data is of multi-input, multi-output type.

    This is currently a proto-type function to get the LodeRunner architecture
    training on a non-variable set of channels.

    Args:
        data (tuple): tuple of model input, corresponding ground truth, and lead time
        model (torch.nn.Module): model to train
        optimizer (torch.optim.Optimizer): optimizer for training set
        loss_fn (torch.nn.Module): loss function for training set
        device (torch.device): device index to select
        channel_map (list): list of channel indices to use

    Returns:
        end_img (torch.Tensor): Ground truth end image
        pred_img (torch.Tensor): Predicted end image
        per_sample_loss (torch.Tensor): Per-sample loss for the batch
    """
    # Set model to train
    model.train()

    # Extract data
    #(start_img, end_img, Dt) = data
    img_seq, Dt = data

    #start_img = start_img.to(device, non_blocking=True)
    #Dt = Dt.to(torch.float32).to(device, non_blocking=True)
    #end_img = end_img.to(device, non_blocking=True)

    img_seq = img_seq.to(device, non_blocking=True)
    Dt = Dt.to(torch.float32).to(device, non_blocking=True)

    start_img = img_seq[:, 0]
    end_img = img_seq[:, -1]

    # For our first LodeRunner training on the lsc240420 dataset the input and
    # output prediction variables are fixed.
    #
    # Both in_vars and out_vars correspond to indices for every variable in
    # this training setup...
    #
    # in_vars = ['density_case',
    #            'density_cushion',
    #            'density_maincharge',
    #            'density_outside_air',
    #            'density_striker',
    #            'density_throw',
    #            'Uvelocity',
    #            'Wvelocity']
    in_vars = torch.tensor(channel_map).to(device, non_blocking=True)
    out_vars = torch.tensor(channel_map).to(device, non_blocking=True)

    # Perform a forward pass
    # NOTE: If training on GPU model should have already been moved to GPU
    # prior to initalizing optimizer.
    print("start_img entering model:", start_img.shape)   # expect [B, 8, 1120, 800]
    print("len(in_vars):", len(in_vars))
    print("in_vars:", in_vars)
    pred_img = model(start_img, in_vars, out_vars, Dt)

    # Expecting to use a *reduction="none"* loss function so we can track loss
    # between individual samples. However, this will make the loss be computed
    # element-wise so we need to still average over the (channel, height,
    # width) dimensions to get the per-sample loss.
    loss = loss_fn(pred_img, end_img)
    per_sample_loss = loss.mean(dim=[1, 2, 3])  # Shape: (batch_size,)

    # Perform backpropagation and update the weights
    optimizer.zero_grad(set_to_none=True)  # Possible speed-up
    loss.mean().backward()
    optimizer.step()

    return end_img, pred_img, per_sample_loss


def train_scheduled_loderunner_datastep(
    data: tuple,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device,
    scheduled_prob: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Trainning step for LodeRunner with scheduled sampling.

    This training step implements scheduled sampling, where the model
    can either use the ground truth image or its own prediction as input.

    Args:
        data (tuple): Sequence of images in (img_seq, Dt) tuple.
        model (loaded pytorch model): model to train.
        optimizer (torch.optim): optimizer for training set.
        loss_fn (torch.nn Loss Function): loss function for training set.
        device (torch.device): device index to select.
        scheduled_prob (float): Probability of using the ground truth as input.

    Returns:
        img_seq (torch.Tensor): Ground truth image sequence.
        pred_seq (torch.Tensor): Predicted image sequence.
        per_sample_loss (torch.Tensor): Per-sample loss for the batch.
    """
    # Set model to train
    model.train()

    # Extract data
    img_seq, Dt = data

    # [B, S, C, H, W] where S=seq-length
    img_seq = img_seq.to(device, non_blocking=True)
    # [B, 1]
    Dt = Dt.to(device, non_blocking=True)

    # Input and output variable indices
    in_vars = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]).to(device, non_blocking=True)
    out_vars = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]).to(device, non_blocking=True)

    # Storage for predictions at each timestep
    pred_seq = []

    # Unbind and iterate over slices in sequence-length dimension
    # NOTE: we exclude img_seq[:, :-1] since we don't have the next
    #   timestep to compare to.
    for k, k_img in enumerate(torch.unbind(img_seq[:, :-1], dim=1)):
        if k == 0:
            # Forward pass for the initial step
            pred_img = model(k_img, in_vars, out_vars, Dt)
        else:
            # Apply scheduled sampling
            if random.random() < scheduled_prob:
                current_input = k_img
            else:
                current_input = pred_img

            pred_img = model(current_input, in_vars, out_vars, Dt)

        # Store the prediction
        pred_seq.append(pred_img)

    # Combine predictions into a tensor of shape [B, SeqLength, C, H, W]
    pred_seq = torch.stack(pred_seq, dim=1)

    # Compute loss
    loss = loss_fn(pred_seq, img_seq[:, 1:])
    per_sample_loss = loss.mean(dim=[1, 2, 3, 4])  # Shape: (batch_size,)

    # Perform backpropagation and update the weights
    optimizer.zero_grad(set_to_none=True)  # Possible speed-up
    loss.mean().backward()
    optimizer.step()

    return img_seq[:, 1:], pred_seq, per_sample_loss


def train_DDP_loderunner_datastep(
    data: tuple,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device,
    rank: int,
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A DDP-compatible training step for multi-input, multi-output data.

    Args:
        data (tuple): tuple of model input, corresponding ground truth, and lead time
        model (loaded pytorch model): model to train
        optimizer (torch.optim): optimizer for training set
        loss_fn (torch.nn Loss Function): loss function for training set
        device (torch.device): device index to select
        rank (int): Rank of device
        world_size (int): Number of total DDP processes

    Returns:
        end_img (torch.Tensor): Ground truth end image
        pred_img (torch.Tensor): Predicted end image
        all_losses (torch.Tensor): Concatenated per-sample losses from all processes
    """
    # Set model to train mode
    model.train()

    # Extract data
    start_img, end_img, Dt = data
    start_img = start_img.to(device, non_blocking=True)
    Dt = Dt.to(device, non_blocking=True)
    end_img = end_img.to(device, non_blocking=True)

    # Fixed input and output variable indices
    in_vars = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]).to(device, non_blocking=True)
    out_vars = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]).to(device, non_blocking=True)

    # Forward pass
    pred_img = model(start_img, in_vars, out_vars, Dt)

    # Compute loss
    loss = loss_fn(pred_img, end_img)
    per_sample_loss = loss.mean(dim=[1, 2, 3])  # Per-sample loss

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

    return end_img, pred_img, all_losses


def train_DDP_loderunner_seq_channel_datastep(
    data,
    model,
    optimizer,
    loss_fn,
    device,
    rank,
    world_size,
):
    model.train()

    start_img, end_img, Dt = data

    start_img = start_img.to(device, non_blocking=True)
    end_img = end_img.to(device, non_blocking=True)
    Dt = Dt.to(torch.float32).to(device, non_blocking=True)

    in_vars = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], device=device)
    out_vars = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], device=device)

    pred_img = model(start_img, in_vars, out_vars, Dt)

    loss = loss_fn(pred_img, end_img)
    per_sample_loss = loss.mean(dim=[1, 2, 3])

    optimizer.zero_grad(set_to_none=True)
    per_sample_loss.mean().backward()
    optimizer.step()

    return end_img, pred_img, per_sample_loss.detach()


def train_DDP_loderunner_seq_datastep(
    data: tuple,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device,
    rank: int,
    world_size: int,
    scheduled_prob: float = 1.0,
    channel_map: list[int] | None = None,
):
    """
    DDP training step for autoregressive sequence training.

    Expected data:
        img_seq, Dt = data

    Shapes:
        img_seq: [B, S, C, H, W]
        Dt:
            either [B] / [B, 1] for a single constant Dt reused at every step,
            or [B, S-1] / [B, S-1, 1] for per-step lead times.

    Returns:
        gt_seq:      [B, S-1, C, H, W]
        pred_seq:    [B, S-1, C, H, W]
        all_losses:  concatenated per-sample losses on rank 0, else None
    """
    model.train()

    img_seq, Dt = data
    img_seq = img_seq.to(device, non_blocking=True)
    Dt = Dt.to(torch.float32).to(device, non_blocking=True)

    if channel_map is None:
        channel_map = [0, 1, 2, 3, 4, 5, 6, 7]

    in_vars = torch.tensor(channel_map, device=device)
    out_vars = torch.tensor(channel_map, device=device)

    B, S, C, H, W = img_seq.shape
    assert S >= 2, "Sequence length must be at least 2."

    pred_seq = []

    # initial input is first frame
    current_input = img_seq[:, 0]

    for k in range(S - 1):
        # support either one Dt for all steps or one Dt per step
        if Dt.ndim == 1 or (Dt.ndim == 2 and Dt.shape[-1] == 1):
            Dt_k = Dt
        elif Dt.ndim == 2:
            Dt_k = Dt[:, k].unsqueeze(-1)
        elif Dt.ndim == 3:
            Dt_k = Dt[:, k]
        else:
            raise ValueError(f"Unsupported Dt shape: {Dt.shape}")

        pred_img = model(current_input, in_vars, out_vars, Dt_k)
        pred_seq.append(pred_img)

        if k < S - 2:
            if random.random() < scheduled_prob:
                current_input = img_seq[:, k + 1]   # teacher forcing
            else:
                current_input = pred_img.detach()   # autoregressive rollout

    pred_seq = torch.stack(pred_seq, dim=1)   # [B, S-1, C, H, W]
    gt_seq = img_seq[:, 1:]                   # [B, S-1, C, H, W]

    loss = loss_fn(pred_seq, gt_seq)
    per_sample_loss = loss.mean(dim=[1, 2, 3, 4])

    optimizer.zero_grad(set_to_none=True)
    loss.mean().backward()
    optimizer.step()

    gathered_losses = [torch.zeros_like(per_sample_loss) for _ in range(world_size)]
    dist.all_gather(gathered_losses, per_sample_loss)

    if rank == 0:
        all_losses = torch.cat(gathered_losses, dim=0)
    else:
        all_losses = None

    return gt_seq, pred_seq, all_losses


def train_DDP_loderunner_datastep_seq_old(
    data: tuple,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device,
    rank: int,
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A DDP-compatible training step for multi-input, multi-output data.

    Args:
        data (tuple): tuple of model input, corresponding ground truth, and lead time
        model (loaded pytorch model): model to train
        optimizer (torch.optim): optimizer for training set
        loss_fn (torch.nn Loss Function): loss function for training set
        device (torch.device): device index to select
        rank (int): Rank of device
        world_size (int): Number of total DDP processes

    Returns:
        end_img (torch.Tensor): Ground truth end image
        pred_img (torch.Tensor): Predicted end image
        all_losses (torch.Tensor): Concatenated per-sample losses from all processes
    """
    # Set model to train mode
    model.train()

    # Extract data
    #start_img, end_img, Dt = data
    img_seq, Dt = data
    #for img in img_seq:
    #    # ...
    start_img = start_img.to(device, non_blocking=True)
    Dt = Dt.to(device, non_blocking=True)
    end_img = end_img.to(device, non_blocking=True)

    # Fixed input and output variable indices
    in_vars = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]).to(device, non_blocking=True)
    out_vars = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]).to(device, non_blocking=True)

    # Forward pass
    pred_img = model(start_img, in_vars, out_vars, Dt)

    # Compute loss
    loss = loss_fn(pred_img, end_img)
    per_sample_loss = loss.mean(dim=[1, 2, 3])  # Per-sample loss

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

    return end_img, pred_img, all_losses


####################################
# Evaluating on a Datastep
####################################
def eval_loderunner_datastep(
    data: tuple,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    device: torch.device,
    channel_map: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """An evaluation step for which the data is of multi-input, multi-output type.

    This is currently a proto-type function to get the LodeRunner architecture
    training on a non-variable set of channels.

    Args:
        data (tuple): tuple of model input, corresponding ground truth, and lead time
        model (torch.nn.Module): model to evaluate
        loss_fn (torch.nn.Module): loss function for evaluation
        device (torch.device): device index to select
        channel_map (list[int]): list of channel indices to use

    Returns:
        end_img (torch.Tensor): Ground truth end image
        pred_img (torch.Tensor): Predicted end image
        per_sample_loss (torch.Tensor): Per-sample loss for the batch

    """
    # Set model to train
    model.eval()

    # Extract data
    (start_img, end_img, Dt) = data
    start_img = start_img.to(device, non_blocking=True)
    Dt = Dt.to(device, non_blocking=True)

    end_img = end_img.to(device, non_blocking=True)

    # For our first LodeRunner training on the lsc240420 dataset the input and
    # output prediction variables are fixed.
    #
    # Both in_vars and out_vars correspond to indices for every variable in
    # this training setup...
    #
    # in_vars = ['density_case',
    #            'density_cushion',
    #            'density_maincharge',
    #            'density_outside_air',
    #            'density_striker',
    #            'density_throw',
    #            'Uvelocity',
    #            'Wvelocity']
    in_vars = torch.tensor(channel_map).to(device, non_blocking=True)
    out_vars = torch.tensor(channel_map).to(device, non_blocking=True)

    # Perform a forward pass
    # NOTE: If training on GPU model should have already been moved to GPU
    # prior to initalizing optimizer.
    pred_img = model(start_img, in_vars, out_vars, Dt)

    # Expecting to use a *reduction="none"* loss function so we can track loss
    # between individual samples. However, this will make the loss be computed
    # element-wise so we need to still average over the (channel, height,
    # width) dimensions to get the per-sample loss.
    loss = loss_fn(pred_img, end_img)
    per_sample_loss = loss.mean(dim=[1, 2, 3])  # Shape: (batch_size,)

    return end_img, pred_img, per_sample_loss


def eval_DDP_loderunner_seq_channel_datastep(
    data,
    model,
    loss_fn,
    device,
    rank,
    world_size,
):
    model.eval()

    start_img, end_img, Dt = data

    start_img = start_img.to(device, non_blocking=True)
    end_img = end_img.to(device, non_blocking=True)
    Dt = Dt.to(torch.float32).to(device, non_blocking=True)

    in_vars = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], device=device)
    out_vars = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], device=device)

    with torch.no_grad():
        pred_img = model(start_img, in_vars, out_vars, Dt)

    loss = loss_fn(pred_img, end_img)
    per_sample_loss = loss.mean(dim=[1, 2, 3])

    # No manual all_gather during eval.
    return end_img, pred_img, per_sample_loss.detach()


def eval_DDP_loderunner_seq_datastep(
    data,
    model,
    loss_fn,
    device,
    rank,
    world_size,
):
    model.eval()

    img_seq, Dt = data
    img_seq = img_seq.to(device, non_blocking=True)
    Dt = Dt.to(torch.float32).to(device, non_blocking=True)

    in_vars = torch.arange(img_seq.shape[2], device=device)
    out_vars = torch.arange(img_seq.shape[2], device=device)

    B, S, C, H, W = img_seq.shape
    current_input = img_seq[:, 0]
    preds = []

    for k in range(S - 1):
        if Dt.ndim == 1:
            Dt_k = Dt
        elif Dt.ndim == 2:
            Dt_k = Dt[:, k]
        else:
            Dt_k = Dt[:, k].squeeze(-1)

        pred_img = model(current_input, in_vars, out_vars, Dt_k)
        preds.append(pred_img)

        if k < S - 2:
            current_input = img_seq[:, k + 1]

    pred_seq = torch.stack(preds, dim=1)
    gt_seq = img_seq[:, 1:]

    loss = loss_fn(pred_seq, gt_seq)
    per_sample_loss = loss.mean(dim=[1, 2, 3, 4])

    gathered_losses = [torch.zeros_like(per_sample_loss) for _ in range(world_size)]
    dist.all_gather(gathered_losses, per_sample_loss)

    if rank == 0:
        all_losses = torch.cat(gathered_losses, dim=0)
    else:
        all_losses = None

    return gt_seq, pred_seq, all_losses


def eval_scheduled_loderunner_datastep(
    data: tuple,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device,
    scheduled_prob: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluation step for LodeRunner with scheduled sampling.

    Args:
        data (tuple): Sequence of images in (img_seq, Dt) tuple.
        model (loaded pytorch model): model to train.
        optimizer (torch.optim): optimizer for training set.
        loss_fn (torch.nn Loss Function): loss function for training set.
        device (torch.device): device index to select.
        scheduled_prob (float): Probability of using the ground truth as input.

    Returns:
        img_seq (torch.Tensor): Ground truth image sequence.
        pred_seq (torch.Tensor): Predicted image sequence.
        per_sample_loss (torch.Tensor): Per-sample loss for the batch.
    """
    # Set model to evaluation
    model.eval()

    # Extract data
    img_seq, Dt = data

    # [B, S, C, H, W] where S=seq-length
    img_seq = img_seq.to(device, non_blocking=True)
    # [B, 1]
    Dt = Dt.to(device, non_blocking=True)

    # Input and output variable indices
    in_vars = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]).to(device, non_blocking=True)
    out_vars = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]).to(device, non_blocking=True)

    # Storage for predictions at each timestep
    pred_seq = []

    # Unbind and iterate over slices in sequence-length dimension
    # NOTE: we exclude img_seq[:, :-1] since we don't have the next
    #   timestep to compare to.
    for k, k_img in enumerate(torch.unbind(img_seq[:, :-1], dim=1)):
        if k == 0:
            # Forward pass for the initial step
            pred_img = model(k_img, in_vars, out_vars, Dt)
        else:
            # Apply scheduled sampling
            if random.random() < scheduled_prob:
                current_input = k_img
            else:
                current_input = pred_img

            pred_img = model(current_input, in_vars, out_vars, Dt)

        # Store the prediction
        pred_seq.append(pred_img)

    # Combine predictions into a tensor of shape [B, SeqLength, C, H, W]
    pred_seq = torch.stack(pred_seq, dim=1)

    # Compute loss
    loss = loss_fn(pred_seq, img_seq[:, 1:])
    per_sample_loss = loss.mean(dim=[1, 2, 3, 4])  # Shape: (batch_size,)

    return img_seq[:, 1:], pred_seq, per_sample_loss


def eval_DDP_loderunner_datastep(
    data: tuple,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    device: torch.device,
    rank: int,
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A DDP-compatible evaluation step.

    Args:
        data (tuple): tuple of model input, corresponding ground truth, and lead time
        model (loaded pytorch model): model to train
        loss_fn (torch.nn Loss Function): loss function for training set
        device (torch.device): device index to select
        rank (int): Rank of device
        world_size (int): Total number of DDP processes

    Returns:
        end_img (torch.Tensor): Ground truth end image
        pred_img (torch.Tensor): Predicted end image
        all_losses (torch.Tensor): Concatenated per-sample losses from all processes
    """
    # Set model to evaluation mode
    model.eval()

    # Extract data
    start_img, end_img, Dt = data
    start_img = start_img.to(device, non_blocking=True)
    Dt = Dt.to(device, non_blocking=True)
    end_img = end_img.to(device, non_blocking=True)

    # Fixed input and output variable indices
    in_vars = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]).to(device, non_blocking=True)
    out_vars = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]).to(device, non_blocking=True)

    # Forward pass
    with torch.no_grad():
        pred_img = model(start_img, in_vars, out_vars, Dt)

    # Compute loss
    loss = loss_fn(pred_img, end_img)
    per_sample_loss = loss.mean(dim=[1, 2, 3])  # Per-sample loss

    # Gather per-sample losses from all processes
    gathered_losses = [torch.zeros_like(per_sample_loss) for _ in range(world_size)]
    dist.all_gather(gathered_losses, per_sample_loss)

    # Rank 0 concatenates and saves or returns all losses
    if rank == 0:
        all_losses = torch.cat(gathered_losses, dim=0)  # Shape: (total_batch_size,)
    else:
        all_losses = None

    return end_img, pred_img, all_losses


def train_DDP_temporal_loderunner_datastep(
    data,
    model,
    optimizer,
    loss_fn,
    device,
    rank,
    world_size,
):
    model.train()

    context_seq, target_img, Dt = data

    context_seq = context_seq.to(device, non_blocking=True)   # [B, K, C, H, W]
    target_img = target_img.to(device, non_blocking=True)     # [B, C, H, W]
    Dt = Dt.to(torch.float32).to(device, non_blocking=True)

    C = target_img.shape[1]
    in_vars = torch.arange(C, device=device)
    out_vars = torch.arange(C, device=device)

    pred_img = model(context_seq, in_vars, out_vars, Dt)

    loss = loss_fn(pred_img, target_img)              # [B, C, H, W] if reduction="none"
    per_sample_loss = loss.mean(dim=[1, 2, 3])

    optimizer.zero_grad(set_to_none=True)
    per_sample_loss.mean().backward()
    optimizer.step()

    gathered_losses = [torch.zeros_like(per_sample_loss) for _ in range(world_size)]
    dist.all_gather(gathered_losses, per_sample_loss)

    if rank == 0:
        all_losses = torch.cat(gathered_losses, dim=0)
    else:
        all_losses = None

    return target_img, pred_img, all_losses


def eval_DDP_temporal_loderunner_datastep(
    data,
    model,
    loss_fn,
    device,
    rank,
    world_size,
):
    model.eval()

    context_seq, target_img, Dt = data

    context_seq = context_seq.to(device, non_blocking=True)
    target_img = target_img.to(device, non_blocking=True)
    Dt = Dt.to(torch.float32).to(device, non_blocking=True)

    C = target_img.shape[1]
    in_vars = torch.arange(C, device=device)
    out_vars = torch.arange(C, device=device)

    pred_img = model(context_seq, in_vars, out_vars, Dt)

    loss = loss_fn(pred_img, target_img)
    per_sample_loss = loss.mean(dim=[1, 2, 3])

    gathered_losses = [torch.zeros_like(per_sample_loss) for _ in range(world_size)]
    dist.all_gather(gathered_losses, per_sample_loss)

    if rank == 0:
        all_losses = torch.cat(gathered_losses, dim=0)
    else:
        all_losses = None

    return target_img, pred_img, all_losses


def eval_DDP_loderunner_seq_context_datastep(
    data,
    model,
    loss_fn,
    device,
    rank,
    world_size,
):
    """
    DDP eval step for TemporalLodeRunner / channel-stacked context model.

    Expected data:
        context_seq, target_img, Dt = data

    Shapes:
        context_seq: [B, K, C, H, W]
        target_img:  [B, C, H, W]
        Dt:          [B] or [B, 1]
    """
    import torch
    import torch.distributed as dist

    model.eval()

    context_seq, target_img, Dt = data

    context_seq = context_seq.to(device, non_blocking=True)
    target_img = target_img.to(device, non_blocking=True)
    Dt = Dt.to(torch.float32).to(device, non_blocking=True)

    C = target_img.shape[1]
    in_vars = torch.arange(C, device=device)
    out_vars = torch.arange(C, device=device)

    pred_img = model(context_seq, in_vars, out_vars, Dt)

    loss = loss_fn(pred_img, target_img)

    if loss.ndim == 0:
        per_sample_loss = loss.repeat(target_img.shape[0])
    else:
        per_sample_loss = loss.mean(dim=tuple(range(1, loss.ndim)))

    gathered_losses = [torch.zeros_like(per_sample_loss) for _ in range(world_size)]
    dist.all_gather(gathered_losses, per_sample_loss)

    if rank == 0:
        all_losses = torch.cat(gathered_losses, dim=0)
    else:
        all_losses = None

    return target_img, pred_img, all_losses
