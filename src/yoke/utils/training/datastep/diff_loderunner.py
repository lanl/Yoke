"""Functions for training and evaluation datasteps for DiffusionLodeRunner.

This module provides functions to perform single training or evaluation steps on a
DiffusionLodeRunner model using denoising score matching objective.
"""

import torch
import torch.distributed as dist

####################################
# Training on a Datastep
####################################
def train_diffusion_loderunner_datastep(
    data: tuple,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device,
    in_vars: torch.Tensor,
    out_vars: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Training step for DiffusionLodeRunner with denoising score matching.

    Args:
        data (tuple): Tuple of (x, y_tau, noise, lead_times, tau) where:
            - x: Conditioning input of shape (B, C_in, H, W)
            - y_tau: Noised target of shape (B, C_out, H, W)
            - noise: Ground truth noise of shape (B, C_out, H, W)
            - lead_times: Lead time values of shape (B,)
            - tau: Diffusion time values of shape (B,) in [0, 1]
        model (torch.nn.Module): DiffusionLodeRunner model to train
        optimizer (torch.optim.Optimizer): Optimizer for training
        loss_fn (torch.nn.Module): Loss function (typically MSE)
        device (torch.device): Device to use for computation
        in_vars (torch.Tensor): Input variable indices for conditioning
        out_vars (torch.Tensor): Output variable indices for prediction

    Returns:
        noise (torch.Tensor): Ground truth noise
        noise_pred (torch.Tensor): Predicted noise
        per_sample_loss (torch.Tensor): Per-sample loss for the batch
    """
    # Set model to train mode
    model.train()

    # Extract data from batch
    x, y_tau, noise, lead_times, tau = data

    # Move data to device
    x = x.to(device, non_blocking=True)
    y_tau = y_tau.to(device, non_blocking=True)
    noise = noise.to(device, non_blocking=True)
    lead_times = lead_times.to(torch.float32).to(device, non_blocking=True)
    tau = tau.to(torch.float32).to(device, non_blocking=True)

    # Make channel indices for all 8 variables (0-7) for both input and output
    vars = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]).to(device, non_blocking=True)

    # Forward pass: predict noise
    noise_pred = model(
        x=x,
        y_tau=y_tau,
        in_vars=vars,
        out_vars=vars,
        lead_times=lead_times,
        diffusion_time=tau,
    )

    # Compute loss (MSE between predicted and true noise)
    # Expecting reduction="none" to track per-sample loss
    loss = loss_fn(noise_pred, noise)
    per_sample_loss = loss.mean(dim=[1, 2, 3])  # Shape: (batch_size,)

    # Backward pass and optimization
    optimizer.zero_grad(set_to_none=True)
    loss.mean().backward()
    optimizer.step()

    return noise, noise_pred, per_sample_loss


def train_DDP_diffusion_loderunner_datastep(
    data: tuple,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device,
    rank: int,
    world_size: int,
    in_vars: torch.Tensor,
    out_vars: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """DDP-compatible training step for DiffusionLodeRunner.

    Args:
        data (tuple): Tuple of (x, y_tau, noise, lead_times, tau)
        model (torch.nn.Module): DiffusionLodeRunner model to train
        optimizer (torch.optim.Optimizer): Optimizer for training
        loss_fn (torch.nn.Module): Loss function (typically MSE)
        device (torch.device): Device to use for computation
        rank (int): Rank of current process
        world_size (int): Total number of DDP processes
        in_vars (torch.Tensor): Input variable indices for conditioning
        out_vars (torch.Tensor): Output variable indices for prediction

    Returns:
        noise (torch.Tensor): Ground truth noise
        noise_pred (torch.Tensor): Predicted noise
        all_losses (torch.Tensor): Concatenated per-sample losses from all processes
    """
    # Set model to train mode
    model.train()

    # Extract data from batch
    x, y_tau, noise, lead_times, tau = data

    # Move data to device
    x = x.to(device, non_blocking=True)
    y_tau = y_tau.to(device, non_blocking=True)
    noise = noise.to(device, non_blocking=True)
    lead_times = lead_times.to(torch.float32).to(device, non_blocking=True)
    tau = tau.to(torch.float32).to(device, non_blocking=True)

    # Make channel indices for all 8 variables (0-7) for both input and output
    vars = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]).to(device, non_blocking=True)

    # Forward pass: predict noise
    noise_pred = model(
        x=x,
        y_tau=y_tau,
        in_vars=vars,
        out_vars=vars,
        lead_times=lead_times,
        diffusion_time=tau,
    )

    # Compute loss
    loss = loss_fn(noise_pred, noise)
    per_sample_loss = loss.mean(dim=[1, 2, 3])  # Per-sample loss

    # Backward pass and optimization
    optimizer.zero_grad(set_to_none=True)
    loss.mean().backward()
    optimizer.step()

    # Gather per-sample losses from all processes
    gathered_losses = [torch.zeros_like(per_sample_loss) for _ in range(world_size)]
    dist.all_gather(gathered_losses, per_sample_loss)

    # Rank 0 concatenates all losses
    if rank == 0:
        all_losses = torch.cat(gathered_losses, dim=0)  # Shape: (total_batch_size,)
    else:
        all_losses = None

    return noise, noise_pred, all_losses


####################################
# Evaluating on a Datastep
####################################
def eval_diffusion_loderunner_datastep(
    data: tuple,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    device: torch.device,
    in_vars: torch.Tensor,
    out_vars: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluation step for DiffusionLodeRunner.

    Args:
        data (tuple): Tuple of (x, y_tau, noise, lead_times, tau)
        model (torch.nn.Module): DiffusionLodeRunner model to evaluate
        loss_fn (torch.nn.Module): Loss function (typically MSE)
        device (torch.device): Device to use for computation
        in_vars (torch.Tensor): Input variable indices for conditioning
        out_vars (torch.Tensor): Output variable indices for prediction

    Returns:
        noise (torch.Tensor): Ground truth noise
        noise_pred (torch.Tensor): Predicted noise
        per_sample_loss (torch.Tensor): Per-sample loss for the batch
    """
    # Set model to evaluation mode
    model.eval()

    # Extract data from batch
    x, y_tau, noise, lead_times, tau = data

    # Move data to device
    x = x.to(device, non_blocking=True)
    y_tau = y_tau.to(device, non_blocking=True)
    noise = noise.to(device, non_blocking=True)
    lead_times = lead_times.to(torch.float32).to(device, non_blocking=True)
    tau = tau.to(torch.float32).to(device, non_blocking=True)

    # Forward pass with no gradient computation
    with torch.no_grad():
        noise_pred = model(
            x=x,
            y_tau=y_tau,
            in_vars=in_vars,
            out_vars=out_vars,
            lead_times=lead_times,
            diffusion_time=tau,
        )

    # Compute loss
    loss = loss_fn(noise_pred, noise)
    per_sample_loss = loss.mean(dim=[1, 2, 3])  # Shape: (batch_size,)

    return noise, noise_pred, per_sample_loss


def eval_DDP_diffusion_loderunner_datastep(
    data: tuple,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    device: torch.device,
    rank: int,
    world_size: int,
    in_vars: torch.Tensor,
    out_vars: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """DDP-compatible evaluation step for DiffusionLodeRunner.

    Args:
        data (tuple): Tuple of (x, y_tau, noise, lead_times, tau)
        model (torch.nn.Module): DiffusionLodeRunner model to evaluate
        loss_fn (torch.nn.Module): Loss function (typically MSE)
        device (torch.device): Device to use for computation
        rank (int): Rank of current process
        world_size (int): Total number of DDP processes
        in_vars (torch.Tensor): Input variable indices for conditioning
        out_vars (torch.Tensor): Output variable indices for prediction

    Returns:
        noise (torch.Tensor): Ground truth noise
        noise_pred (torch.Tensor): Predicted noise
        all_losses (torch.Tensor): Concatenated per-sample losses from all processes
    """
    # Set model to evaluation mode
    model.eval()

    # Extract data from batch
    x, y_tau, noise, lead_times, tau = data

    # Move data to device
    x = x.to(device, non_blocking=True)
    y_tau = y_tau.to(device, non_blocking=True)
    noise = noise.to(device, non_blocking=True)
    lead_times = lead_times.to(torch.float32).to(device, non_blocking=True)
    tau = tau.to(torch.float32).to(device, non_blocking=True)

    # Forward pass with no gradient computation
    with torch.no_grad():
        noise_pred = model(
            x=x,
            y_tau=y_tau,
            in_vars=in_vars,
            out_vars=out_vars,
            lead_times=lead_times,
            diffusion_time=tau,
        )

    # Compute loss
    loss = loss_fn(noise_pred, noise)
    per_sample_loss = loss.mean(dim=[1, 2, 3])  # Per-sample loss

    # Gather per-sample losses from all processes
    gathered_losses = [torch.zeros_like(per_sample_loss) for _ in range(world_size)]
    dist.all_gather(gathered_losses, per_sample_loss)

    # Rank 0 concatenates all losses
    if rank == 0:
        all_losses = torch.cat(gathered_losses, dim=0)  # Shape: (total_batch_size,)
    else:
        all_losses = None

    return noise, noise_pred, all_losses


################################################
# Checking that the Data & Model Work Together
################################################
if __name__ == "__main__":
    """Test the diffusion datastep functions with real dataset."""
    import argparse
    import sys
    from pathlib import Path
    import numpy as np

    # Add parent directory to path for imports
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

    from yoke.models.vit.swin.diffusion_bomberman import DiffusionLodeRunner
    from yoke.datasets.diffusion_dataset import DiffusionLSC_temporal_DataSet

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Test DiffusionLodeRunner datastep functions with real data"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing NPZ files",
    )
    parser.add_argument(
        "--file_prefix_list",
        type=str,
        required=True,
        help="Text file with list of file prefixes",
    )
    parser.add_argument(
        "--max_timeIDX_offset",
        type=int,
        default=10,
        help="Maximum time index offset (default: 10)",
    )
    parser.add_argument(
        "--max_file_checks",
        type=int,
        default=100,
        help="Maximum file check attempts (default: 100)",
    )
    parser.add_argument(
        "--half_image",
        action="store_true",
        help="Use half images (no reflection)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Testing DiffusionLodeRunner Datastep Functions")
    print("=" * 60)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Define variables for LSC dataset
    in_vars = np.array([
        "density_case",
        "density_cushion",
        "density_maincharge",
        "density_outside_air",
        "density_striker",
        "density_throw",
        "Uvelocity",
        "Wvelocity",
    ])
    out_vars = in_vars  # Same variables for input and output

    # Create dataset
    print("\nCreating DiffusionLSC_temporal_DataSet...")
    dataset = DiffusionLSC_temporal_DataSet(
        LSC_NPZ_DIR=args.data_dir,
        file_prefix_list=args.file_prefix_list,
        max_timeIDX_offset=args.max_timeIDX_offset,
        max_file_checks=args.max_file_checks,
        half_image=args.half_image,
        in_vars=in_vars,
        out_vars=out_vars,
    )
    print(f"Dataset created with {dataset.Nsamples} file prefixes")

    # Get a sample to determine image dimensions
    print("\nLoading sample to determine dimensions...")
    data = dataset[0]
    sample_x, sample_y_tau, sample_noise, sample_lead_time, sample_tau = data
    height, width = sample_x.shape[1], sample_x.shape[2]
    in_channels = sample_x.shape[0]
    out_channels = sample_y_tau.shape[0]
    print(f"Image dimensions: {height}x{width}")
    print(f"Input channels: {in_channels}")
    print(f"Output channels: {out_channels}")

    # Convert to indicies for training
    in_vars_ch = torch.tensor(list(range(in_channels)))
    out_vars_ch = torch.tensor(list(range(out_channels)))

    print("\nCreating DiffusionLodeRunner model...")
    model = DiffusionLodeRunner(
        default_vars=list(in_vars),
        image_size=(height, width),
        patch_size=(10, 10),
        embed_dim=96,
        emb_factor=2,
        num_heads=8,
        block_structure=(1, 1, 3, 1),
        window_sizes=[(8, 8), (8, 8), (4, 4), (2, 2)],
        patch_merge_scales=[(2, 2), (2, 2), (2, 2)],
        verbose=False,
    ).to(device)
    print("Model created successfully")

    # Create optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.MSELoss(reduction="none")

    # Test training step
    print("\n" + "-" * 60)
    print("Testing train_diffusion_loderunner_datastep")
    print("-" * 60)

    data_batch = tuple(item.unsqueeze(0) for item in data)
    noise, noise_pred, per_sample_loss = train_diffusion_loderunner_datastep(
        data=data_batch,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        in_vars=in_vars_ch,
        out_vars=out_vars_ch,
    )

    print(f"Ground truth noise shape: {noise.shape}")
    print(f"Predicted noise shape: {noise_pred.shape}")
    print(f"Per-sample loss shape: {per_sample_loss.shape}")
    print(f"Mean loss: {per_sample_loss.mean().item():.6f}")

    # Test evaluation step
    print("\n" + "-" * 60)
    print("Testing eval_diffusion_loderunner_datastep")
    print("-" * 60)

    noise, noise_pred, per_sample_loss = eval_diffusion_loderunner_datastep(
        data=data_batch,
        model=model,
        loss_fn=loss_fn,
        device=device,
        in_vars=in_vars_ch,
        out_vars=out_vars_ch,
    )

    print(f"Ground truth noise shape: {noise.shape}")
    print(f"Predicted noise shape: {noise_pred.shape}")
    print(f"Per-sample loss shape: {per_sample_loss.shape}")
    print(f"Mean loss: {per_sample_loss.mean().item():.6f}")

    print("\nTesting completed successfully.")
