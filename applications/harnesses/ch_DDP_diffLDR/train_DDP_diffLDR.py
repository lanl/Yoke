"""DDP training script for DiffusionLodeRunner.

This script trains the DiffusionLodeRunner model using distributed data parallel (DDP)
on the LSC dataset with score-based diffusion modeling.
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from yoke.models.vit.swin.diffusion_bomberman import DiffusionLodeRunner
from yoke.datasets.diffusion_dataset import DiffusionLSC_temporal_DataSet
from yoke.utils.diffusion.noise_schedulers import VPCosineNoiseSchedule
from yoke.utils.training.epoch.diff_loderunner import train_DDP_diffusion_loderunner_epoch
from yoke.harnesses.base import HarnessStudy
from yoke.utils.dataload import make_distributed_dataloader
from yoke.utils.checkpointing import load_model_and_optimizer
from yoke.utils.checkpointing import save_model_and_optimizer
from yoke.utils.parallel import setup_distributed, cleanup_distributed
from yoke.lr_schedulers import CosineWithWarmupScheduler
from yoke.helpers import cli


#############################################
# Inputs
#############################################
descr_str = (
    "Uses DDP to train DiffusionLodeRunner architecture on temporal prediction "
    "of the lsc240420 per-material density fields using score-clearbased diffusion."
)
parser = argparse.ArgumentParser(
    prog="DDP DiffusionLodeRunner Training",
    description=descr_str,
    fromfile_prefix_chars="@",
)
parser = cli.add_default_args(parser=parser)
parser = cli.add_filepath_args(parser=parser)
parser = cli.add_computing_args(parser=parser)
parser = cli.add_model_args(parser=parser)
parser = cli.add_training_args(parser=parser)
parser = cli.add_cosine_lr_scheduler_args(parser=parser)

# Diffusion-specific parameters
parser.add_argument(
    "--max_timeIDX_offset",
    type=int,
    default=10,
    help="Maximum time offset for input/output image pairs.",
)

# Change some default filepaths
parser.set_defaults(
    train_filelist="lsc240420_prefixes_train_80pct.txt",
    validation_filelist="lsc240420_prefixes_validation_10pct.txt",
    test_filelist="lsc240420_prefixes_test_10pct.txt",
)


def main(args, rank, world_size, local_rank, device):
    """Main training function."""
    #############################################
    # Process Inputs
    #############################################
    studyIDX = args.studyIDX
    Ngpus = args.Ngpus
    Knodes = args.Knodes

    # Data Paths
    train_filelist = args.FILELIST_DIR + args.train_filelist
    validation_filelist = args.FILELIST_DIR + args.validation_filelist

    # Model Parameters
    embed_dim = args.embed_dim
    block_structure = tuple(args.block_structure)

    # Training parameters
    max_timeIDX_offset = args.max_timeIDX_offset
    num_workers = args.num_workers
    batch_size = args.batch_size
    total_epochs = args.total_epochs
    cycle_epochs = args.cycle_epochs
    train_batches = args.train_batches
    val_batches = args.val_batches
    train_per_val = args.TRAIN_PER_VAL
    trn_rcrd_filename = args.trn_rcrd_filename
    val_rcrd_filename = args.val_rcrd_filename
    CONTINUATION = args.continuation
    checkpoint = args.checkpoint

    #############################################
    # Model Arguments
    #############################################
    # Dictionary of available models.
    available_models = {
        "DiffusionLodeRunner": DiffusionLodeRunner
    }

    # Define channels
    channel_list = [
        "density_case",
        #"energy_case",
        #"pressure_case",
        "density_cushion",
        #"energy_cushion",
        #"pressure_cushion",
        "density_maincharge",
        #"energy_maincharge",
        #"pressure_maincharge",
        "density_outside_air",
        #"energy_outside_air",
        #"pressure_outside_air",
        "density_striker",
        #"energy_striker",
        #"pressure_striker",
        "density_throw",
        #"energy_throw",
        #"pressure_throw",
        "Uvelocity",
        "Wvelocity",
    ]

    # Model arguments for DiffusionLodeRunner
    model_args = {
        "default_vars": channel_list,
        "image_size": (1120, 400),
        "patch_size": (10, 10),
        "embed_dim": embed_dim,
        "emb_factor": 2,
        "num_heads": 8,
        "block_structure": block_structure,
        "window_sizes": [(2, 2), (2, 2), (2, 2), (2, 2)],
        "patch_merge_scales": [(2, 2), (2, 2), (2, 2)],
    }

    # Variable indices (using all channels for both input and output)
    in_vars = torch.tensor(list(range(len(channel_list)))).to(device)
    out_vars = torch.tensor(list(range(len(channel_list)))).to(device)

    #############################################
    # Initialize Noise Schedule
    #############################################
    noise_schedule = VPCosineNoiseSchedule()

    #############################################
    # Load Model for Continuation
    #############################################
    if CONTINUATION:
        model, optimizer, starting_epoch = load_model_and_optimizer(
            checkpoint,
            optimizer_class=torch.optim.AdamW,
            optimizer_kwargs={
                "lr": 1e-4,
                "betas": (0.9, 0.999),
                "eps": 1e-08,
                "weight_decay": 0.01,
            },
            available_models=available_models,
            device=device,
        )
        if rank == 0:
            print("Model state loaded for continuation.")
    else:
        starting_epoch = 0
        model = DiffusionLodeRunner(**model_args)
        model.to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0.01,
        )

        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)

    #############################################
    # Initialize Loss
    #############################################
    loss_fn = nn.MSELoss(reduction="none")

    #############################################
    # Move Model to DistributedDataParallel
    #############################################
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    #############################################
    # Learning Rate Scheduler
    #############################################
    if rank == 0:
        print(f"Starting epoch: {starting_epoch}")

    if starting_epoch == 0:
        last_epoch = -1
    else:
        last_epoch = train_batches * (starting_epoch - 1)

    LRsched = CosineWithWarmupScheduler(
        optimizer,
        warmup_steps=args.warmup_steps,
        anchor_lr=args.anchor_lr,
        terminal_steps=args.terminal_steps,
        num_cycles=args.num_cycles,
        min_fraction=args.min_fraction,
        last_epoch=last_epoch,
    )

    #############################################
    # Data Initialization
    #############################################
    train_dataset = DiffusionLSC_temporal_DataSet(
        LSC_NPZ_DIR=args.LSC_NPZ_DIR,
        file_prefix_list=train_filelist,
        max_timeIDX_offset=max_timeIDX_offset,
        max_file_checks=10,
        half_image=True,
        in_vars=np.array(channel_list),
        out_vars=np.array(channel_list),
        noise_schedule=noise_schedule,
    )

    val_dataset = DiffusionLSC_temporal_DataSet(
        LSC_NPZ_DIR=args.LSC_NPZ_DIR,
        file_prefix_list=validation_filelist,
        max_timeIDX_offset=max_timeIDX_offset,
        max_file_checks=10,
        half_image=True,
        in_vars=np.array(channel_list),
        out_vars=np.array(channel_list),
        noise_schedule=noise_schedule,
    )

    train_dataloader = make_distributed_dataloader(
        train_dataset,
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        rank=rank,
        world_size=world_size,
    )

    val_dataloader = make_distributed_dataloader(
        val_dataset,
        batch_size,
        shuffle=False,
        num_workers=num_workers,
        rank=rank,
        world_size=world_size,
    )

    #############################################
    # Training Loop
    #############################################
    if rank == 0:
        print("Training Model . . .")

    starting_epoch += 1
    ending_epoch = min(starting_epoch + cycle_epochs, total_epochs + 1)

    TIME_EPOCH = True
    for epochIDX in range(starting_epoch, ending_epoch):
        train_sampler = train_dataloader.sampler
        train_sampler.set_epoch(epochIDX)

        if TIME_EPOCH:
            dist.barrier()
            torch.cuda.synchronize(device)
            startTime = time.time()

        # Train and Validate
        train_DDP_diffusion_loderunner_epoch(
            training_data=train_dataloader,
            validation_data=val_dataloader,
            num_train_batches=train_batches,
            num_val_batches=val_batches,
            model=model,
            in_vars=in_vars,
            out_vars=out_vars,
            optimizer=optimizer,
            loss_fn=loss_fn,
            LRsched=LRsched,
            epochIDX=epochIDX,
            train_per_val=train_per_val,
            train_rcrd_filename=trn_rcrd_filename,
            val_rcrd_filename=val_rcrd_filename,
            device=device,
            rank=rank,
            world_size=world_size,
        )

        if TIME_EPOCH:
            torch.cuda.synchronize(device)
            dist.barrier()
            endTime = time.time()

        epoch_time = (endTime - startTime) / 60

        if rank == 0:
            print(f"Completed epoch {epochIDX}...", flush=True)
            print(f"Epoch time (minutes): {epoch_time:.2f}", flush=True)

    # Save model and optimizer
    if rank == 0:
        chkpt_name_str = f"study{studyIDX:03d}_modelState_epoch{epochIDX:04d}.pth"
        new_chkpt_path = os.path.join("./", chkpt_name_str)

        save_model_and_optimizer(
            model.module,  # Save the underlying model, not the DDP wrapper
            optimizer,
            epochIDX,
            new_chkpt_path,
            model_class=DiffusionLodeRunner,
            model_args=model_args,
        )

        #############################################
        # Continue if Necessary
        #############################################
        FINISHED_TRAINING = epochIDX + 1 > total_epochs
        if not FINISHED_TRAINING:
            new_slurm_file = HarnessStudy.continuation_setup(
                new_chkpt_path, studyIDX, last_epoch=epochIDX
            )
            os.system(f"sbatch {new_slurm_file}")


if __name__ == "__main__":
    args = parser.parse_args()

    rank, world_size, local_rank, device = setup_distributed()

    main(args, rank, world_size, local_rank, device)

    cleanup_distributed()
