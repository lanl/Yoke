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
    band_weights: torch.Tensor = None,
    ema: object = None,
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

    ``band_weights`` (optional [n_bands] tensor) up-weights the backward pass
    per observed band. Because targets are per-band z-scored, equal weighting
    lets large-dynamic-range bands (u, g) contribute little to the loss and be
    under-fit; weighting scales each sample by its observed band's weight. The
    RECORDED per-sample loss stays unweighted so the CSV metric is comparable
    across runs. ``None`` (default) reproduces the plain equal-weight behavior.

    ``ema`` (optional :class:`yoke.utils.ema.ParamEMA`) shadows the trainable
    params and is updated after each ``optimizer.step()`` (Polyak averaging);
    ``None`` (default) is a no-op.
    """
    train_rcrd_filename = train_rcrd_filename.replace(
        "<epochIDX>",
        f"{epochIDX:04d}",
    )

    model.train()

    if band_weights is not None:
        band_weights = band_weights.to(device)

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

            if band_weights is None:
                # Recorded metric == training objective: plain equal weight.
                batch_loss = per_sample_loss.mean()
            else:
                # Weight the backward by the observed band's weight (mask is
                # one-hot, so this picks each sample's band weight). The
                # recorded per_sample_loss above stays unweighted so the CSV
                # metric is comparable across runs and weightings.
                sample_w = (mask * band_weights.reshape(1, -1)).sum(dim=1)
                batch_loss = (per_sample_loss * sample_w).sum() / (
                    sample_w.sum() + 1e-8
                )

            batch_loss.backward()
            optimizer.step()
            LRsched.step()

            if ema is not None:
                core = model.module if hasattr(model, "module") else model
                ema.update(
                    (n, p) for n, p in core.named_parameters() if p.requires_grad
                )

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
    band_weights: torch.Tensor = None,
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
        band_weights (torch.Tensor): Optional per-band gradient weights
            [n_bands]. When given, ``total_loss`` (the backward objective) is a
            per-band weighted mean over valid steps; ``per_sample_loss`` (the
            recorded metric) stays unweighted. ``None`` reproduces the plain
            equal-weight behavior exactly.

    Returns:
        per_sample_loss (torch.Tensor): Mean rollout loss per sample [B].
        total_loss (torch.Tensor): Scalar mean loss over all valid steps
            (per-band weighted when ``band_weights`` is given).
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
    step_bands = []  # [B] observed band index per step (for band weighting)

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
        step_bands.append(tgt_band)

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

    if band_weights is None:
        total_loss = step_losses.sum() / (step_valid.sum() + 1e-8)
    else:
        # Per-band-weighted objective: scale each valid step by its observed
        # band's weight. per_sample_loss (recorded) stays unweighted above.
        step_bands = torch.stack(step_bands, dim=1)  # [B, n_steps]
        bw = band_weights.to(device)
        step_w = bw[step_bands] * step_valid  # [B, n_steps]
        total_loss = (step_losses * bw[step_bands]).sum() / (
            step_w.sum() + 1e-8
        )

    return per_sample_loss, total_loss


def _rollout_pass_9band_window(
    ctx_v: torch.Tensor,
    ctx_t: torch.Tensor,
    ctx_b: torch.Tensor,
    ctx_valid: torch.Tensor,
    future_v: torch.Tensor,
    future_b: torch.Tensor,
    future_dt: torch.Tensor,
    future_valid: torch.Tensor,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    n_bands: int,
    context_window_days: float,
    max_context_len: int,
    teacher_forcing_ratio: float,
    device: torch.device,
    band_weights: torch.Tensor = None,
    dt_weight_tau: float = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unroll the 9-band model over a batch of rollouts in time-window mode.

    The time-window analogue of :func:`_rollout_pass_9band`. Instead of a
    fixed-count window slid by drop-oldest, this keeps a **growing** buffer of
    absolute-time events and, at each step, re-selects the trailing
    ``context_window_days`` window (capped to the most recent
    ``max_context_len``), padded with a per-event validity flag. This mirrors the
    inference rollout (``get_rollout_from_stream`` + ``_select_window`` in
    ``plot_pred_diagnostics_9band.py``) so the model is trained on exactly the
    context layout it is evaluated on: per-event width ``3 + n_bands``
    (``[value, rel_t, valid, one_hot]``), ``rel_t`` relative to the window's
    first real event.

    Because the buffer is time-sorted and the window is "events within W days of
    the most recent event, capped to the most recent ``max_context_len``", the
    selected window is always a contiguous suffix, so re-selection is a batched
    start-index + gather (no per-row Python loop, no scatter).

    Args:
        ctx_v (torch.Tensor): Padded seed context values [B, seed_width].
        ctx_t (torch.Tensor): Padded seed ABSOLUTE context times [B, seed_width].
        ctx_b (torch.Tensor): Padded seed context band indices [B, seed_width].
        ctx_valid (torch.Tensor): Seed validity mask [B, seed_width].
        future_v (torch.Tensor): True future values [B, n_rollout_steps].
        future_b (torch.Tensor): Future band indices [B, n_rollout_steps].
        future_dt (torch.Tensor): Future lead times [B, n_rollout_steps].
        future_valid (torch.Tensor): Valid-step mask [B, n_rollout_steps].
        model (torch.nn.Module): The 9-band wrapper model.
        loss_fn (torch.nn.Module): Elementwise loss (reduction='none').
        n_bands (int): Number of bands.
        context_window_days (float): Trailing lookback W in days.
        max_context_len (int): Padded context width M.
        teacher_forcing_ratio (float): Probability of feeding the true value back
            at each step (1.0 = fully teacher-forced, 0.0 = fully free-running).
        device (torch.device): Compute device.
        band_weights (torch.Tensor): Optional per-band gradient weights
            [n_bands]. When given, ``total_loss`` (the backward objective) is a
            per-band weighted mean over valid steps; ``per_sample_loss`` (the
            recorded metric) stays unweighted. ``None`` reproduces the plain
            equal-weight behavior exactly.
        dt_weight_tau (float): Optional lead-time weighting time-constant in days.
            When set, each rollout step is weighted by ``1 / (1 + Dt / tau)`` in the
            backward objective, so short-lead steps (the early rise, where the model
            fails) get relatively more gradient than the many slowly-fading
            late-tail steps (which otherwise dominate a per-point mean and pull the
            forecast toward a flat plateau). Composes multiplicatively with
            ``band_weights``. ``per_sample_loss`` (the recorded metric) stays
            unweighted. ``None`` (default) disables it, reproducing prior behavior.

    Returns:
        per_sample_loss (torch.Tensor): Mean rollout loss per sample [B].
        total_loss (torch.Tensor): Scalar mean loss over all valid steps
            (per-band weighted when ``band_weights`` is given).
    """
    B = ctx_v.shape[0]
    seed_width = ctx_v.shape[1]
    n_steps = future_v.shape[1]
    W = float(context_window_days)
    M = int(max_context_len)

    # Growing buffer: the seed (<= M real events, left-packed) plus at most one
    # appended event per rollout step. Left-packed and time-sorted throughout.
    C = M + n_steps

    buf_v = torch.zeros(B, C, device=device, dtype=ctx_v.dtype)
    buf_t = torch.zeros(B, C, device=device, dtype=ctx_t.dtype)
    buf_b = torch.zeros(B, C, device=device, dtype=torch.long)

    buf_v[:, :seed_width] = ctx_v
    buf_t[:, :seed_width] = ctx_t
    buf_b[:, :seed_width] = ctx_b

    # Number of real events currently in each row's buffer (>= 1 in window mode,
    # since the anchor event is always inside its own trailing window).
    count = ctx_valid.sum(dim=1).long()  # [B]

    batch_arange = torch.arange(B, device=device)
    pos = torch.arange(C, device=device).unsqueeze(0)  # [1, C]
    out_pos = torch.arange(M, device=device).unsqueeze(0)  # [1, M]

    # Kept for API compatibility with the LodeRunner-style wrapper.
    in_vars = torch.arange(8, device=device)
    out_vars = torch.arange(8, device=device)

    step_losses = []  # [B] per step
    step_valid = []  # [B] per step
    step_bands = []  # [B] observed band index per step (for band weighting)

    for step in range(n_steps):
        # Most recent real event time per row (left-packed => index count - 1).
        last_idx = (count - 1).clamp(min=0)
        last_t = buf_t.gather(1, last_idx.unsqueeze(1)).squeeze(1)  # [B]
        lo = last_t - W

        # Trailing W-day window is a contiguous suffix of the sorted buffer.
        real_mask = pos < count.unsqueeze(1)  # [B, C]
        ge_mask = buf_t >= lo.unsqueeze(1)  # [B, C]
        n_in_window = (real_mask & ge_mask).sum(dim=1)  # [B]
        win_len = torch.clamp(n_in_window, max=M)  # [B]
        start = count - win_len  # [B]

        # Source buffer index for each padded output position p in [0, M).
        src_idx = start.unsqueeze(1) + out_pos  # [B, M]
        valid_out = out_pos < win_len.unsqueeze(1)  # [B, M] bool
        src_idx_c = src_idx.clamp(max=C - 1)

        gathered_v = buf_v.gather(1, src_idx_c)  # [B, M]
        gathered_t = buf_t.gather(1, src_idx_c)  # [B, M]
        gathered_b = buf_b.gather(1, src_idx_c)  # [B, M]

        # rel_t relative to the window's first real event (buf_t[start]).
        first_t = buf_t.gather(1, start.clamp(max=C - 1).unsqueeze(1))  # [B, 1]
        rel_t = gathered_t - first_t  # [B, M]

        valid_f = valid_out.to(buf_v.dtype)  # [B, M]
        val_col = gathered_v * valid_f
        rel_col = rel_t * valid_f

        band_onehot = torch.zeros(
            B, M, n_bands, device=device, dtype=buf_v.dtype
        )
        band_onehot.scatter_(2, gathered_b.unsqueeze(-1), 1.0)
        band_onehot = band_onehot * valid_f.unsqueeze(-1)

        per_event = torch.cat(
            [
                val_col.unsqueeze(-1),
                rel_col.unsqueeze(-1),
                valid_f.unsqueeze(-1),
                band_onehot,
            ],
            dim=-1,
        )  # [B, M, 3 + n_bands]
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
        step_bands.append(tgt_band)

        # Scheduled sampling: choose true vs own (detached) prediction per sample.
        use_true = torch.rand(B, device=device) < teacher_forcing_ratio
        fed = torch.where(use_true, true_obs, pred_obs.detach())

        # Append the new event (following the true time/band schedule) at the
        # left-packed write position and grow the count. For padded steps the
        # appended event is spurious but harmless: those steps' losses are masked
        # and all later steps for that row are padded too. The W-day re-selection
        # at the next step handles dropping stale events (never drop-oldest here),
        # matching the inference rollout which appends to its running context.
        new_t = last_t + Dt

        write_pos = count.clamp(max=C - 1).unsqueeze(1)  # [B, 1]
        buf_v.scatter_(1, write_pos, fed.unsqueeze(1))
        buf_t.scatter_(1, write_pos, new_t.unsqueeze(1))
        buf_b.scatter_(1, write_pos, tgt_band.unsqueeze(1))
        count = torch.clamp(count + 1, max=C)

    step_losses = torch.stack(step_losses, dim=1)  # [B, n_steps]
    step_valid = torch.stack(step_valid, dim=1)  # [B, n_steps]

    per_sample_loss = step_losses.sum(dim=1) / (step_valid.sum(dim=1) + 1e-8)

    if band_weights is None and dt_weight_tau is None:
        total_loss = step_losses.sum() / (step_valid.sum() + 1e-8)
    else:
        # Weighted objective: scale each valid step by its observed band's weight
        # (band_weights) and/or a lead-time weight that down-weights long horizons
        # (dt_weight_tau). Both compose multiplicatively; per_sample_loss
        # (recorded) stays unweighted above.
        step_w = step_valid.clone()  # [B, n_steps]
        if band_weights is not None:
            step_bands = torch.stack(step_bands, dim=1)  # [B, n_steps]
            step_w = step_w * band_weights.to(device)[step_bands]
        if dt_weight_tau is not None:
            # 1 / (1 + Dt / tau): 1.0 at Dt=0, decaying with lead time. future_dt
            # is [B, n_steps]; clamp Dt>=0 so padded/degenerate steps stay sane.
            dt_w = 1.0 / (1.0 + future_dt.clamp_min(0.0) / float(dt_weight_tau))
            step_w = step_w * dt_w
        total_loss = (step_losses * step_w).sum() / (step_w.sum() + 1e-8)

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
    window_mode: bool = False,
    context_window_days: float = None,
    max_context_len: int = None,
    band_weights: torch.Tensor = None,
    dt_weight_tau: float = None,
    ema: object = None,
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
        fixed-count mode (``window_mode=False``), 7-tuple:
            ctx_v, ctx_t, ctx_b:  [B, context_len]
            future_v, future_b, future_dt, future_valid:  [B, n_rollout_steps]
        time-window mode (``window_mode=True``), 8-tuple (adds ``ctx_valid``):
            ctx_v, ctx_t, ctx_b, ctx_valid:  [B, max_context_len]
            future_v, future_b, future_dt, future_valid:  [B, n_rollout_steps]

    Args:
        n_bands (int): Number of bands the model predicts.
        teacher_forcing_ratio (float): Per-epoch probability of feeding the true
            value back at each rollout step during training.
        window_mode (bool): If True, consume the 8-tuple time-window sample and
            unroll with the trailing-W-day context re-selection
            (:func:`_rollout_pass_9band_window`). If False (default), the legacy
            fixed-count 7-tuple path (:func:`_rollout_pass_9band`).
        context_window_days (float): Trailing lookback W in days. Required when
            ``window_mode`` is True.
        max_context_len (int): Padded context width M. Required when
            ``window_mode`` is True.
        band_weights (torch.Tensor): Optional per-band gradient weights
            [n_bands], forwarded to the rollout pass to up-weight
            large-dynamic-range bands (u, g) in the backward objective. The
            recorded per-sample loss stays unweighted. ``None`` (default)
            reproduces the plain equal-weight behavior. Validation always runs
            unweighted so the recorded metric is comparable.
        dt_weight_tau (float): Optional lead-time weighting time-constant in days
            (window mode only), forwarded to the rollout pass to down-weight
            long-horizon steps so the early rise is not swamped by the many
            late-tail points. The recorded per-sample loss stays unweighted, and
            validation always runs unweighted. ``None`` (default) disables it.
        ema (object): Optional :class:`yoke.utils.ema.ParamEMA` shadow of the
            trainable params (conditioner + output_head). When supplied, its
            ``update`` is called after every ``optimizer.step()`` so the average
            tracks the per-batch trajectory (Polyak averaging). Only the
            ``requires_grad`` params of the underlying (DDP-unwrapped) module are
            shadowed. ``None`` (default) is a no-op.
    """
    def _run_pass(
        data: tuple[torch.Tensor, ...],
        ratio: float,
        weights: torch.Tensor = None,
        dt_tau: float = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Unpack a batch (7- or 8-tuple), move to device, run the rollout."""
        if window_mode:
            (
                ctx_v,
                ctx_t,
                ctx_b,
                ctx_valid,
                future_v,
                future_b,
                future_dt,
                future_valid,
            ) = data
            ctx_valid = ctx_valid.to(device, non_blocking=True)
        else:
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
        future_dt = future_dt.to(torch.float32).to(device, non_blocking=True)
        future_valid = future_valid.to(device, non_blocking=True)

        if window_mode:
            return _rollout_pass_9band_window(
                ctx_v=ctx_v,
                ctx_t=ctx_t,
                ctx_b=ctx_b,
                ctx_valid=ctx_valid,
                future_v=future_v,
                future_b=future_b,
                future_dt=future_dt,
                future_valid=future_valid,
                model=model,
                loss_fn=loss_fn,
                n_bands=n_bands,
                context_window_days=context_window_days,
                max_context_len=max_context_len,
                teacher_forcing_ratio=ratio,
                device=device,
                band_weights=weights,
                dt_weight_tau=dt_tau,
            )

        return _rollout_pass_9band(
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
            teacher_forcing_ratio=ratio,
            device=device,
            band_weights=weights,
        )

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

            optimizer.zero_grad(set_to_none=True)

            per_sample_loss, batch_loss = _run_pass(
                data, teacher_forcing_ratio, weights=band_weights, dt_tau=dt_weight_tau
            )

            batch_loss.backward()
            optimizer.step()
            LRsched.step()

            if ema is not None:
                core = model.module if hasattr(model, "module") else model
                ema.update(
                    (n, p) for n, p in core.named_parameters() if p.requires_grad
                )

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

                    # Validation is always a pure free-running rollout.
                    per_sample_loss, _ = _run_pass(data, 0.0)

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
