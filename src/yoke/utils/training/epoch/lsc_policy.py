"""Functions to train and evaluate an lsc240420 policy network over a single epoch."""

import math
import torch
import numpy as np
from contextlib import nullcontext
from collections.abc import Callable
from typing import Optional

from yoke.utils.training.datastep.lsc_policy import (
    train_lsc_policy_datastep,
    eval_lsc_policy_datastep,
)


def train_lsc_policy_epoch(
    training_data: torch.utils.data.DataLoader,
    validation_data: torch.utils.data.DataLoader,
    num_train_batches: int,
    num_val_batches: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    LRsched: torch.optim.lr_scheduler._LRScheduler,
    epochIDX: int,
    train_per_val: int,
    train_rcrd_filename: str,
    val_rcrd_filename: str,
    device: torch.device,
    rank: int,
    world_size: int,
    blocks: Optional[list[tuple[str, Callable[[str], bool]]]] = None,
) -> None:
    """Epoch training of LSC Gaussian-policy network.

    Function to complete a training epoch on the LSC Gaussian-policy network for the
    layered shaped charge design problem. Training and validation information
    is saved to successive CSV files.

    Args:
        training_data (torch.utils.data.DataLoader): training dataloader
        validation_data (torch.utils.data.DataLoader): validation dataloader
        num_train_batches (int): Number of batches in training epoch
        num_val_batches (int): Number of batches in validation epoch
        model (torch.nn.Module): model to train
        optimizer (torch.optim.Optimizer): optimizer for training set
        loss_fn (torch.nn.Module): loss function for training set
        LRsched (torch.optim.lr_scheduler._LRScheduler): Learning-rate scheduler called
                                                         every training step.
        epochIDX (int): Index of current training epoch
        train_per_val (int): Number of Training epochs between each validation
        train_rcrd_filename (str): Name of CSV file to save training sample stats to
        val_rcrd_filename (str): Name of CSV file to save validation sample stats to
        device (torch.device): device index to select
        rank (int): rank of process
        world_size (int): number of total processes
        blocks (list[tuple[str, Callable[[str], bool]]]): (OPTIONAL) List of unfrozen
                                                          blocks in network for gradient
                                                          observation.
    """
    # Initialize things to save
    trainbatch_ID = 0
    valbatch_ID = 0

    # Training loop
    model.train()
    train_rcrd_filename = train_rcrd_filename.replace("<epochIDX>", f"{epochIDX:04d}")

    # Add condition if there is a blocks argument to monitor gradients
    BLOCK_CND = blocks is not None
    if BLOCK_CND:
        grad_rcrd_filename = train_rcrd_filename.replace(".csv", "_grad.csv")

    with (
        open(train_rcrd_filename, "a") if rank == 0 else nullcontext()
    ) as train_rcrd_file, (
        open(grad_rcrd_filename, "a") if (rank == 0 and BLOCK_CND) else nullcontext()
    ) as grad_rcrd_file:
        # Write header to training record file (rank 0 only)
        if rank == 0:
            trn_header = ["epoch", "batch", "loss"]
            np.savetxt(train_rcrd_file, [trn_header], fmt="%s", delimiter=",")
            train_rcrd_file.flush()

            if BLOCK_CND:
                grad_header = ["epoch", "batch"] + [name for name, _ in blocks]
                np.savetxt(grad_rcrd_file, [grad_header], fmt="%s", delimiter=",")
                grad_rcrd_file.flush()

        # Iterate over training batches
        for trainbatch_ID, traindata in enumerate(training_data):
            # Stop when number of training batches is reached
            if trainbatch_ID >= num_train_batches:
                break

            # Perform a single training step
            x_true, pred_mean, train_losses = train_lsc_policy_datastep(
                traindata, model, optimizer, loss_fn, device, rank, world_size,
            )

            # Increment the learning-rate scheduler
            LRsched.step()

            # Save training record (rank 0 only)
            if rank == 0:
                # Save RMS-gradients for unfrozen blocks
                if BLOCK_CND:
                    rms_list = []
                    for blk_name, blk_match in blocks:
                        total_sq = 0.0
                        total_count = 0
                        for n, p in model.module.named_parameters():
                            if p.requires_grad and blk_match(n) and p.grad is not None:
                                g = p.grad.detach().view(-1)
                                total_sq += float((g * g).sum().item())
                                total_count += g.numel()

                        if total_count > 0:
                            rms = math.sqrt(total_sq / total_count)
                        else:
                            rms = 0.0
                        rms_list.append(rms)

                # build a (1, N) array so numpy.savetxt writes a single row
                row = np.array(
                    [epochIDX, trainbatch_ID] + rms_list,
                    dtype=float
                    )[None, :]
                fmt = ["%d", "%d"] + ["%.10f"] * len(rms_list)
                np.savetxt(grad_rcrd_file, row, fmt=fmt, delimiter=",")

                # Save training losses
                batch_records = np.column_stack(
                    [
                        np.full(len(train_losses), epochIDX),
                        np.full(len(train_losses), trainbatch_ID),
                        train_losses.cpu().numpy().flatten(),
                    ]
                )
                np.savetxt(train_rcrd_file, batch_records, fmt="%d, %d, %.8f")

    # Validation loop
    if epochIDX % train_per_val == 0:
        print("Validating...", epochIDX)
        val_rcrd_filename = val_rcrd_filename.replace("<epochIDX>", f"{epochIDX:04d}")
        model.eval()
        with (
            open(val_rcrd_filename, "a") if rank == 0 else nullcontext()
        ) as val_rcrd_file:
            with torch.no_grad():
                for valbatch_ID, valdata in enumerate(validation_data):
                    # Stop when number of training batches is reached
                    if valbatch_ID >= num_val_batches:
                        break

                    x_true, pred_mean, val_losses = eval_lsc_policy_datastep(
                        valdata,
                        model,
                        loss_fn,
                        device,
                        rank,
                        world_size,
                    )

                    # Save validation record (rank 0 only)
                    if rank == 0:
                        batch_records = np.column_stack(
                            [
                                np.full(len(val_losses), epochIDX),
                                np.full(len(val_losses), valbatch_ID),
                                val_losses.cpu().numpy().flatten(),
                            ]
                        )
                        np.savetxt(val_rcrd_file, batch_records, fmt="%d, %d, %.8f")
