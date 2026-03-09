"""Functions to train and evaluate LodeRunner over a single epoch."""

import torch
import numpy as np
import time
from contextlib import nullcontext

from yoke.utils.training.datastep.loderunner import (
    train_loderunner_datastep,
    eval_loderunner_datastep,
    train_scheduled_loderunner_datastep,
    eval_scheduled_loderunner_datastep,
    train_DDP_loderunner_datastep,
    eval_DDP_loderunner_datastep,
)


def train_simple_loderunner_epoch(
    channel_map: list,
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
    verbose: bool = False,
) -> None:
    """Training and validation epochs on the LodeRunner architecture.

    Training and validation information is saved to successive CSV files.

    Args:
        channel_map (list): List mapping input/output channels for the model.
        training_data (torch.utils.data.DataLoader): training dataloader
        validation_data (torch.utils.data.DataLoader): validation dataloader
        model (torch.nn.Module): model to train
        optimizer (torch.optim.Optimizer): optimizer for training set
        loss_fn (torch.nn.Module): loss function for training set
        epochIDX (int): Index of current training epoch
        train_per_val (int): Number of Training epochs between each validation
        train_rcrd_filename (str): Name of CSV file to save training sample stats to
        val_rcrd_filename (str): Name of CSV file to save validation sample stats to
        device (torch.device): device index to select
        verbose (bool): Flag to print diagnostic output.
    """
    # Initialize things to save
    trainbatch_ID = 0
    valbatch_ID = 0

    train_batchsize = training_data.batch_size
    val_batchsize = validation_data.batch_size

    train_rcrd_filename = train_rcrd_filename.replace("<epochIDX>", f"{epochIDX:04d}")
    # Train on all training samples
    with open(train_rcrd_filename, "a") as train_rcrd_file:
        for traindata in training_data:
            trainbatch_ID += 1

            # Time each epoch and print to stdout
            if verbose:
                startTime = time.time()

            truth, pred, train_loss = train_loderunner_datastep(
                traindata, model, optimizer, loss_fn, device, channel_map
            )

            if verbose:
                endTime = time.time()
                batch_time = endTime - startTime
                print(
                    f"Batch {trainbatch_ID} time (seconds): {batch_time:.5f}", flush=True
                )

            if verbose:
                startTime = time.time()

            # Stack loss record and write using numpy
            batch_records = np.column_stack(
                [
                    np.full(train_batchsize, epochIDX),
                    np.full(train_batchsize, trainbatch_ID),
                    train_loss.detach().cpu().numpy().flatten(),
                ]
            )

            np.savetxt(train_rcrd_file, batch_records, fmt="%d, %d, %.8f")

            if verbose:
                endTime = time.time()
                record_time = endTime - startTime
                print(
                    f"Batch {trainbatch_ID} record time: {record_time:.5f}", flush=True
                )

    # Evaluate on all validation samples
    if epochIDX % train_per_val == 0:
        print("Validating...", epochIDX)
        val_rcrd_filename = val_rcrd_filename.replace("<epochIDX>", f"{epochIDX:04d}")
        with open(val_rcrd_filename, "a") as val_rcrd_file:
            with torch.no_grad():
                for valdata in validation_data:
                    valbatch_ID += 1
                    truth, pred, val_loss = eval_loderunner_datastep(
                        valdata, model, loss_fn, device, channel_map
                    )

                    # Stack loss record and write using numpy
                    batch_records = np.column_stack(
                        [
                            np.full(val_batchsize, epochIDX),
                            np.full(val_batchsize, valbatch_ID),
                            val_loss.detach().cpu().numpy().flatten(),
                        ]
                    )

                    np.savetxt(val_rcrd_file, batch_records, fmt="%d, %d, %.8f")


def train_scheduled_loderunner_epoch(
    training_data: torch.utils.data.DataLoader,
    validation_data: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    LRsched: torch.optim.lr_scheduler._LRScheduler,
    loss_fn: torch.nn.Module,
    epochIDX: int,
    train_per_val: int,
    train_rcrd_filename: str,
    val_rcrd_filename: str,
    device: torch.device,
    scheduled_prob: float,
) -> float:
    """Training and validation epoch for LodeRunner architecture with scheduled sampling.

    Updates the scheduled probability over time.

    Args:
        training_data (torch.utils.data.DataLoader): training dataloader
        validation_data (torch.utils.data.DataLoader): validation dataloader
        model (torch.nn.Module): model to train.
        optimizer (torch.optim.Optimizer): optimizer for training set.
        LRsched (torch.optim.lr_scheduler._LRScheduler): Learning-rate scheduler called
                                                         every training step.
        loss_fn (torch.nn.Module): loss function for training set.
        epochIDX (int): Index of current training epoch.
        train_per_val (int): Number of training epochs between each validation.
        train_rcrd_filename (str): Name of CSV file to save training sample stats to.
        val_rcrd_filename (str): Name of CSV file to save validation sample stats to.
        device (torch.device): device index to select.
        scheduled_prob (float): Initial probability of using ground truth as input.

    Returns:
        scheduled_prob (float): Updated scheduled probability for the next epoch.
    """
    # Initialize variables for tracking batches
    trainbatch_ID = 0
    valbatch_ID = 0

    train_rcrd_filename = train_rcrd_filename.replace("<epochIDX>", f"{epochIDX:04d}")

    # Train on all training samples
    with open(train_rcrd_filename, "a") as train_rcrd_file:
        for traindata in training_data:
            trainbatch_ID += 1

            # Training step with scheduled sampling
            true_seq, pred_seq, train_losses = train_scheduled_loderunner_datastep(
                data=traindata,
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                device=device,
                scheduled_prob=scheduled_prob,
            )

            # Increment the learning-rate scheduler
            LRsched.step()

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
                    truth, pred, val_losses = eval_scheduled_loderunner_datastep(
                        data=valdata,
                        model=model,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        device=device,
                        scheduled_prob=scheduled_prob,
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

    # Return the updated scheduled probability
    return scheduled_prob


def train_LRsched_loderunner_epoch(
    channel_map: list,
    training_data: torch.utils.data.DataLoader,
    validation_data: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochIDX: int,
    LRsched: torch.optim.lr_scheduler._LRScheduler,
    train_per_val: int,
    train_rcrd_filename: str,
    val_rcrd_filename: str,
    device: torch.device,
    verbose: bool = False,
) -> None:
    """Training and validation epoch on LodeRunner with LR-scheduler.

    Args:
        channel_map (list): List mapping input/output channels for the model.
        training_data (torch.utils.data.DataLoader): training dataloader
        validation_data (torch.utils.data.DataLoader): validation dataloader
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
        verbose (bool): Flag to print diagnostic output.
    """
    # Initialize things to save
    trainbatch_ID = 0
    valbatch_ID = 0

    train_batchsize = training_data.batch_size
    val_batchsize = validation_data.batch_size

    train_rcrd_filename = train_rcrd_filename.replace("<epochIDX>", f"{epochIDX:04d}")
    # Train on all training samples
    with open(train_rcrd_filename, "a") as train_rcrd_file:
        for traindata in training_data:
            trainbatch_ID += 1

            # Time each epoch and print to stdout
            if verbose:
                startTime = time.time()

            truth, pred, train_loss = train_loderunner_datastep(
                traindata, model, optimizer, loss_fn, device, channel_map
            )

            # Increment the learning-rate scheduler
            LRsched.step()

            if verbose:
                endTime = time.time()
                batch_time = endTime - startTime
                print(
                    f"Batch {trainbatch_ID} time (seconds): {batch_time:.5f}", flush=True
                )

            if verbose:
                startTime = time.time()

            # Stack loss record and write using numpy
            batch_records = np.column_stack(
                [
                    np.full(train_batchsize, epochIDX),
                    np.full(train_batchsize, trainbatch_ID),
                    train_loss.detach().cpu().numpy().flatten(),
                ]
            )

            np.savetxt(train_rcrd_file, batch_records, fmt="%d, %d, %.8f")

            if verbose:
                endTime = time.time()
                record_time = endTime - startTime
                print(
                    f"Batch {trainbatch_ID} record time: {record_time:.5f}", flush=True
                )

    # Evaluate on all validation samples
    if epochIDX % train_per_val == 0:
        print("Validating...", epochIDX)
        val_rcrd_filename = val_rcrd_filename.replace("<epochIDX>", f"{epochIDX:04d}")
        with open(val_rcrd_filename, "a") as val_rcrd_file:
            with torch.no_grad():
                for valdata in validation_data:
                    valbatch_ID += 1
                    truth, pred, val_loss = eval_loderunner_datastep(
                        valdata, model, loss_fn, device, channel_map
                    )

                    # Stack loss record and write using numpy
                    batch_records = np.column_stack(
                        [
                            np.full(val_batchsize, epochIDX),
                            np.full(val_batchsize, valbatch_ID),
                            val_loss.detach().cpu().numpy().flatten(),
                        ]
                    )

                    np.savetxt(val_rcrd_file, batch_records, fmt="%d, %d, %.8f")


def train_DDP_loderunner_epoch(
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
) -> None:
    """Distributed data-parallel LodeRunner Epoch.

    Function to complete a training epoch on the LodeRunner architecture with
    fixed channels in the input and output. Training and validation information
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

    """
    # Initialize things to save
    trainbatch_ID = 0
    valbatch_ID = 0

    # Training loop
    model.train()
    train_rcrd_filename = train_rcrd_filename.replace("<epochIDX>", f"{epochIDX:04d}")
    with (
        open(train_rcrd_filename, "a") if rank == 0 else nullcontext()
    ) as train_rcrd_file:
        for trainbatch_ID, traindata in enumerate(training_data):
            # Stop when number of training batches is reached
            if trainbatch_ID >= num_train_batches:
                break

            # Perform a single training step
            truth, pred, train_losses = train_DDP_loderunner_datastep(
                traindata, model, optimizer, loss_fn, device, rank, world_size
            )

            # Increment the learning-rate scheduler
            LRsched.step()

            # Save training record (rank 0 only)
            if rank == 0:
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

                    end_img, pred_img, val_losses = eval_DDP_loderunner_datastep(
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
