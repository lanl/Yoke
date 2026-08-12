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


def train_DDP_scalar_temporal_loderunner_epoch_gri(
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
    """DDP epoch function for scalar temporal LodeRunner training.

    Expected dataset output:
        x:      [B, input_dim]
        target: [B, n_outputs]
        Dt:     [B]

    For the 3-band kilonova case:
        x:      [B, 4 * context_len]
                flattened [g, r, i] context plus relative times
        target: [B, 3]
                normalized delta_g, delta_r, delta_i
        Dt:     [B]

    Expected model output:
        pred:   [B, 3]
    """
    train_rcrd_filename = train_rcrd_filename.replace(
        "<epochIDX>",
        f"{epochIDX:04d}",
    )

    model.train()

    with (
        open(train_rcrd_filename, "a") if rank == 0 else nullcontext()
    ) as train_rcrd_file:

        for trainbatch_ID, data in enumerate(training_data):
            if trainbatch_ID >= num_train_batches:
                break

            x, target, Dt = data

            x = x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            Dt = Dt.to(torch.float32).to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # These are kept for API compatibility with LodeRunner-style wrappers.
            # ScalarTemporalConditionedLodeRunner may ignore them internally,
            # or pass them to the backbone.
            in_vars = torch.arange(8, device=device)
            out_vars = torch.arange(8, device=device)

            pred = model(x, in_vars, out_vars, Dt)

            if pred.shape != target.shape:
                raise RuntimeError(
                    f"Prediction and target shapes do not match: "
                    f"pred.shape={pred.shape}, target.shape={target.shape}"
                )

            loss = loss_fn(pred, target)

            # Huber/MSE with reduction='none' gives [B, 3].
            # Reduce over output channels, leaving one loss per sample.
            if loss.ndim == 1:
                per_sample_loss = loss
            else:
                per_sample_loss = loss.mean(dim=tuple(range(1, loss.ndim)))

            batch_loss = per_sample_loss.mean()

            batch_loss.backward()
            optimizer.step()
            LRsched.step()

            if rank == 0:
                batch_records = np.column_stack(
                    [
                        np.full(len(per_sample_loss), epochIDX),
                        np.full(len(per_sample_loss), trainbatch_ID),
                        per_sample_loss.detach().cpu().numpy().flatten(),
                    ]
                )
                np.savetxt(train_rcrd_file, batch_records, fmt="%d, %d, %.8f")

    if epochIDX % train_per_val == 0:
        if rank == 0:
            print("Validating...", epochIDX, flush=True)

        val_rcrd_filename = val_rcrd_filename.replace(
            "<epochIDX>",
            f"{epochIDX:04d}",
        )

        model.eval()

        with (
            open(val_rcrd_filename, "a") if rank == 0 else nullcontext()
        ) as val_rcrd_file:

            with torch.no_grad():
                for valbatch_ID, data in enumerate(validation_data):
                    if valbatch_ID >= num_val_batches:
                        break

                    x, target, Dt = data

                    x = x.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)
                    Dt = Dt.to(torch.float32).to(device, non_blocking=True)

                    in_vars = torch.arange(8, device=device)
                    out_vars = torch.arange(8, device=device)

                    pred = model(x, in_vars, out_vars, Dt)

                    if pred.shape != target.shape:
                        raise RuntimeError(
                            f"Validation prediction and target shapes do not match: "
                            f"pred.shape={pred.shape}, target.shape={target.shape}"
                        )

                    loss = loss_fn(pred, target)

                    if loss.ndim == 1:
                        per_sample_loss = loss
                    else:
                        per_sample_loss = loss.mean(dim=tuple(range(1, loss.ndim)))

                    if rank == 0:
                        batch_records = np.column_stack(
                            [
                                np.full(len(per_sample_loss), epochIDX),
                                np.full(len(per_sample_loss), valbatch_ID),
                                per_sample_loss.detach().cpu().numpy().flatten(),
                            ]
                        )
                        np.savetxt(val_rcrd_file, batch_records, fmt="%d, %d, %.8f")


def train_DDP_scalar_temporal_loderunner_epoch_9band(
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
    """DDP epoch function for the masked 9-band scalar temporal LodeRunner.

    The dataloader yields a merged event-stream context plus a masked, per-band
    target. The model predicts all bands; the loss is masked to the single band
    observed at the target event.

    Expected dataset output:
        x:      [B, context_len * (2 + n_bands)]
        target: [B, n_bands]   normalized value, only observed band meaningful
        mask:   [B, n_bands]   1.0 for observed band, 0.0 elsewhere
        Dt:     [B]

    Expected model output:
        pred:   [B, n_bands]
    """
    train_rcrd_filename = train_rcrd_filename.replace(
        "<epochIDX>",
        f"{epochIDX:04d}",
    )

    model.train()

    with (
        open(train_rcrd_filename, "a") if rank == 0 else nullcontext()
    ) as train_rcrd_file:

        for trainbatch_ID, data in enumerate(training_data):
            if trainbatch_ID >= num_train_batches:
                break

            x, target, mask, Dt = data

            x = x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            Dt = Dt.to(torch.float32).to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Kept for API compatibility with LodeRunner-style wrappers.
            in_vars = torch.arange(8, device=device)
            out_vars = torch.arange(8, device=device)

            pred = model(x, in_vars, out_vars, Dt)

            if pred.shape != target.shape:
                raise RuntimeError(
                    f"Prediction and target shapes do not match: "
                    f"pred.shape={pred.shape}, target.shape={target.shape}"
                )

            # loss_fn uses reduction='none' -> [B, n_bands]. Mask to the
            # observed band and average per sample over observed entries.
            loss = loss_fn(pred, target) * mask
            per_sample_loss = loss.sum(dim=1) / (mask.sum(dim=1) + 1e-8)

            batch_loss = per_sample_loss.mean()

            batch_loss.backward()
            optimizer.step()
            LRsched.step()

            if rank == 0:
                batch_records = np.column_stack(
                    [
                        np.full(len(per_sample_loss), epochIDX),
                        np.full(len(per_sample_loss), trainbatch_ID),
                        per_sample_loss.detach().cpu().numpy().flatten(),
                    ]
                )
                np.savetxt(train_rcrd_file, batch_records, fmt="%d, %d, %.8f")

    if epochIDX % train_per_val == 0:
        if rank == 0:
            print("Validating...", epochIDX, flush=True)

        val_rcrd_filename = val_rcrd_filename.replace(
            "<epochIDX>",
            f"{epochIDX:04d}",
        )

        model.eval()

        with (
            open(val_rcrd_filename, "a") if rank == 0 else nullcontext()
        ) as val_rcrd_file:

            with torch.no_grad():
                for valbatch_ID, data in enumerate(validation_data):
                    if valbatch_ID >= num_val_batches:
                        break

                    x, target, mask, Dt = data

                    x = x.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)
                    mask = mask.to(device, non_blocking=True)
                    Dt = Dt.to(torch.float32).to(device, non_blocking=True)

                    in_vars = torch.arange(8, device=device)
                    out_vars = torch.arange(8, device=device)

                    pred = model(x, in_vars, out_vars, Dt)

                    if pred.shape != target.shape:
                        raise RuntimeError(
                            f"Validation prediction and target shapes do not "
                            f"match: pred.shape={pred.shape}, "
                            f"target.shape={target.shape}"
                        )

                    loss = loss_fn(pred, target) * mask
                    per_sample_loss = loss.sum(dim=1) / (mask.sum(dim=1) + 1e-8)

                    if rank == 0:
                        batch_records = np.column_stack(
                            [
                                np.full(len(per_sample_loss), epochIDX),
                                np.full(len(per_sample_loss), valbatch_ID),
                                per_sample_loss.detach().cpu().numpy().flatten(),
                            ]
                        )
                        np.savetxt(val_rcrd_file, batch_records, fmt="%d, %d, %.8f")


def _rollout_pass_9band(
    ctx_v: torch.Tensor,
    ctx_t: torch.Tensor,
    ctx_b: torch.Tensor,
    future_v: torch.Tensor,
    future_b: torch.Tensor,
    future_dt: torch.Tensor,
    future_valid: torch.Tensor,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    n_bands: int,
    teacher_forcing_ratio: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unroll the 9-band model over a batch of rollouts with scheduled sampling.

    Maintains a batched sliding context window seeded from the initial context.
    At each step the model predicts all bands at the true lead time; the loss is
    taken on the single observed band (padding steps ignored via
    ``future_valid``). The value fed back into the context is the true value with
    probability ``teacher_forcing_ratio`` and the model's own (detached)
    prediction otherwise, so gradients never flow through the rollout.

    Args:
        ctx_v (torch.Tensor): Initial context values [B, context_len].
        ctx_t (torch.Tensor): Initial context relative times [B, context_len].
        ctx_b (torch.Tensor): Initial context band indices [B, context_len].
        future_v (torch.Tensor): True future values [B, n_rollout_steps].
        future_b (torch.Tensor): Future band indices [B, n_rollout_steps].
        future_dt (torch.Tensor): Future lead times [B, n_rollout_steps].
        future_valid (torch.Tensor): Valid-step mask [B, n_rollout_steps].
        model (torch.nn.Module): The 9-band wrapper model.
        loss_fn (torch.nn.Module): Elementwise loss (reduction='none').
        n_bands (int): Number of bands.
        teacher_forcing_ratio (float): Probability of feeding the true value back
            at each step (1.0 = fully teacher-forced, 0.0 = fully free-running).
        device (torch.device): Compute device.

    Returns:
        per_sample_loss (torch.Tensor): Mean rollout loss per sample [B].
        total_loss (torch.Tensor): Scalar mean loss over all valid steps.
    """
    B, context_len = ctx_v.shape
    n_steps = future_v.shape[1]

    # Running window; cloned so we can slide in-place without touching inputs.
    win_v = ctx_v.clone()
    win_t = ctx_t.clone()
    win_b = ctx_b.clone()

    batch_arange = torch.arange(B, device=device)

    # Kept for API compatibility with the LodeRunner-style wrapper.
    in_vars = torch.arange(8, device=device)
    out_vars = torch.arange(8, device=device)

    step_losses = []  # [B] per valid step
    step_valid = []  # [B] per step

    for step in range(n_steps):
        # Build the flattened per-event input from the current window, with
        # times made relative to the window's first event (matching the dataset
        # encoding and the inference rollout).
        rel_t = win_t - win_t[:, :1]
        band_onehot = torch.zeros(
            B, context_len, n_bands, device=device, dtype=win_v.dtype
        )
        band_onehot.scatter_(2, win_b.unsqueeze(-1), 1.0)

        per_event = torch.cat(
            [win_v.unsqueeze(-1), rel_t.unsqueeze(-1), band_onehot],
            dim=-1,
        )  # [B, context_len, 2 + n_bands]
        x_step = per_event.reshape(B, -1)

        Dt = future_dt[:, step]
        pred_all = model(x_step, in_vars, out_vars, Dt)  # [B, n_bands]

        tgt_band = future_b[:, step]
        pred_obs = pred_all[batch_arange, tgt_band]  # [B]
        true_obs = future_v[:, step]  # [B]
        valid = future_valid[:, step]  # [B]

        step_loss = loss_fn(pred_obs, true_obs) * valid
        step_losses.append(step_loss)
        step_valid.append(valid)

        # Scheduled sampling: choose true vs own (detached) prediction per sample.
        use_true = (
            torch.rand(B, device=device) < teacher_forcing_ratio
        )
        fed = torch.where(use_true, true_obs, pred_obs.detach())

        # For padded steps there is no real event to advance to; feeding the true
        # (zero) value with a zero dt is harmless since their loss is masked out
        # and later steps are also padded/masked.
        new_t = win_t[:, -1] + Dt

        # Slide the window: drop the oldest event, append the new one.
        win_v = torch.cat([win_v[:, 1:], fed.unsqueeze(1)], dim=1)
        win_t = torch.cat([win_t[:, 1:], new_t.unsqueeze(1)], dim=1)
        win_b = torch.cat([win_b[:, 1:], tgt_band.unsqueeze(1)], dim=1)

    step_losses = torch.stack(step_losses, dim=1)  # [B, n_steps]
    step_valid = torch.stack(step_valid, dim=1)  # [B, n_steps]

    per_sample_loss = step_losses.sum(dim=1) / (step_valid.sum(dim=1) + 1e-8)
    total_loss = step_losses.sum() / (step_valid.sum() + 1e-8)

    return per_sample_loss, total_loss


def train_DDP_scalar_temporal_loderunner_epoch_9band_rollout(
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
    n_bands: int = 9,
    teacher_forcing_ratio: float = 1.0,
) -> None:
    """Multi-step rollout DDP epoch for the masked 9-band scalar temporal model.

    Trains the model on autoregressive rollouts with scheduled sampling to
    address exposure bias: instead of a single teacher-forced next-event step,
    the model unrolls ``n_rollout_steps`` events, feeding its own (detached)
    predictions back into the context with probability
    ``1 - teacher_forcing_ratio`` at each step. Validation always uses a fully
    free-running rollout (ratio 0.0) so the recorded metric reflects real
    rollout skill.

    Expected dataset output (per sample, from
    ``Kilonova_lc_scalar_context_DataSet_9band`` with ``n_rollout_steps > 1``):
        ctx_v, ctx_t, ctx_b:  [B, context_len]
        future_v, future_b, future_dt, future_valid:  [B, n_rollout_steps]

    Args:
        n_bands (int): Number of bands the model predicts.
        teacher_forcing_ratio (float): Per-epoch probability of feeding the true
            value back at each rollout step during training.
    """
    train_rcrd_filename = train_rcrd_filename.replace(
        "<epochIDX>",
        f"{epochIDX:04d}",
    )

    model.train()

    with (
        open(train_rcrd_filename, "a") if rank == 0 else nullcontext()
    ) as train_rcrd_file:

        for trainbatch_ID, data in enumerate(training_data):
            if trainbatch_ID >= num_train_batches:
                break

            ctx_v, ctx_t, ctx_b, future_v, future_b, future_dt, future_valid = (
                data
            )

            ctx_v = ctx_v.to(device, non_blocking=True)
            ctx_t = ctx_t.to(device, non_blocking=True)
            ctx_b = ctx_b.to(device, non_blocking=True)
            future_v = future_v.to(device, non_blocking=True)
            future_b = future_b.to(device, non_blocking=True)
            future_dt = future_dt.to(torch.float32).to(device, non_blocking=True)
            future_valid = future_valid.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            per_sample_loss, batch_loss = _rollout_pass_9band(
                ctx_v=ctx_v,
                ctx_t=ctx_t,
                ctx_b=ctx_b,
                future_v=future_v,
                future_b=future_b,
                future_dt=future_dt,
                future_valid=future_valid,
                model=model,
                loss_fn=loss_fn,
                n_bands=n_bands,
                teacher_forcing_ratio=teacher_forcing_ratio,
                device=device,
            )

            batch_loss.backward()
            optimizer.step()
            LRsched.step()

            if rank == 0:
                batch_records = np.column_stack(
                    [
                        np.full(len(per_sample_loss), epochIDX),
                        np.full(len(per_sample_loss), trainbatch_ID),
                        per_sample_loss.detach().cpu().numpy().flatten(),
                    ]
                )
                np.savetxt(train_rcrd_file, batch_records, fmt="%d, %d, %.8f")

    if epochIDX % train_per_val == 0:
        if rank == 0:
            print("Validating...", epochIDX, flush=True)

        val_rcrd_filename = val_rcrd_filename.replace(
            "<epochIDX>",
            f"{epochIDX:04d}",
        )

        model.eval()

        with (
            open(val_rcrd_filename, "a") if rank == 0 else nullcontext()
        ) as val_rcrd_file:

            with torch.no_grad():
                for valbatch_ID, data in enumerate(validation_data):
                    if valbatch_ID >= num_val_batches:
                        break

                    (
                        ctx_v,
                        ctx_t,
                        ctx_b,
                        future_v,
                        future_b,
                        future_dt,
                        future_valid,
                    ) = data

                    ctx_v = ctx_v.to(device, non_blocking=True)
                    ctx_t = ctx_t.to(device, non_blocking=True)
                    ctx_b = ctx_b.to(device, non_blocking=True)
                    future_v = future_v.to(device, non_blocking=True)
                    future_b = future_b.to(device, non_blocking=True)
                    future_dt = future_dt.to(torch.float32).to(
                        device, non_blocking=True
                    )
                    future_valid = future_valid.to(device, non_blocking=True)

                    # Validation is always a pure free-running rollout.
                    per_sample_loss, _ = _rollout_pass_9band(
                        ctx_v=ctx_v,
                        ctx_t=ctx_t,
                        ctx_b=ctx_b,
                        future_v=future_v,
                        future_b=future_b,
                        future_dt=future_dt,
                        future_valid=future_valid,
                        model=model,
                        loss_fn=loss_fn,
                        n_bands=n_bands,
                        teacher_forcing_ratio=0.0,
                        device=device,
                    )

                    if rank == 0:
                        batch_records = np.column_stack(
                            [
                                np.full(len(per_sample_loss), epochIDX),
                                np.full(len(per_sample_loss), valbatch_ID),
                                per_sample_loss.detach().cpu().numpy().flatten(),
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
