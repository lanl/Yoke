"""Functions to run a training and validation epoch for a network with array outputs."""

import torch
import numpy as np

from yoke.utils.training.datastep.array_output import (
    train_array_datastep,
    eval_array_datastep,
)


def train_array_epoch(
    training_data: torch.utils.data.DataLoader,
    validation_data: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochIDX: int,
    train_per_val: int,
    train_rcrd_filename: str,
    val_rcrd_filename: str,
    device: torch.device,
) -> None:
    """Epoch for a network which has an array as output.

    Training and validation information is saved to successive CSV files.

    Args:
        training_data (torch.dataloader): dataloader containing the training samples
        validation_data (torch.dataloader): dataloader containing the validation samples
        model (loaded pytorch model): model to train
        optimizer (torch.optim): optimizer for training set
        loss_fn (torch.nn Loss Function): loss function for training set
        epochIDX (int): Index of current training epoch
        train_per_val (int): Number of Training epochs between each validation
        train_rcrd_filename (str): Name of CSV file to save training sample stats to
        val_rcrd_filename (str): Name of CSV file to save validation sample stats to
        device (torch.device): device index to select

    """
    trainbatch_ID = 0
    valbatch_ID = 0

    train_rcrd_filename = train_rcrd_filename.replace("<epochIDX>", f"{epochIDX:04d}")
    # Train on all training samples
    with open(train_rcrd_filename, "a") as train_rcrd_file:
        for traindata in training_data:
            trainbatch_ID += 1
            truth, pred, train_losses = train_array_datastep(
                traindata, model, optimizer, loss_fn, device
            )

            # Save batch records to the training record file
            batch_records = np.column_stack(
                [
                    np.full(len(train_losses), epochIDX),
                    np.full(len(train_losses), trainbatch_ID),
                    train_losses.detach().cpu().numpy().flatten(),
                ]
            )
            np.savetxt(train_rcrd_file, batch_records, fmt="%d, %d, %.8f")

    # Evaluate on all validation samples
    if epochIDX % train_per_val == 0:
        print("Validating...", epochIDX)
        val_rcrd_filename = val_rcrd_filename.replace("<epochIDX>", f"{epochIDX:04d}")
        with open(val_rcrd_filename, "a") as val_rcrd_file:
            with torch.no_grad():
                for valdata in validation_data:
                    valbatch_ID += 1
                    truth, pred, val_losses = eval_array_datastep(
                        valdata, model, loss_fn, device
                    )

                    # Save validation batch records
                    batch_records = np.column_stack(
                        [
                            np.full(len(val_losses), epochIDX),
                            np.full(len(val_losses), valbatch_ID),
                            val_losses.detach().cpu().numpy().flatten(),
                        ]
                    )
                    np.savetxt(val_rcrd_file, batch_records, fmt="%d, %d, %.8f")
