"""Train a Gaussian Policy network using DDP."""

import os
import time
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from yoke.models.policyCNNmodules import gaussian_policyCNN
from yoke.datasets.lsc_dataset import LSC_hfield_policy_DataSet
import yoke.torch_training_utils as tr
from yoke.helpers import cli


#############################################
# Inputs
#############################################
descr_str = (
    "Uses DDP to train Gaussian policy architecture."
)
parser = argparse.ArgumentParser(
    prog="Gaussian Policy Training", description=descr_str, fromfile_prefix_chars="@"
)
parser = cli.add_default_args(parser=parser)
parser = cli.add_filepath_args(parser=parser)
parser = cli.add_training_args(parser=parser)


def setup_distributed() -> tuple[int, int, int, torch.device]:
    """Sets up distributed training using PyTorch DDP."""
    # ----- 1) Basic setup & environment variables -----
    # Rely on Slurm variables: SLURM_PROCID, SLURM_NTASKS, SLURM_LOCALID, etc.
    rank = int(os.environ["SLURM_PROCID"])  # global rank
    world_size = int(os.environ["SLURM_NTASKS"])  # total number of processes
    local_rank = int(os.environ["SLURM_LOCALID"])  # local rank (GPU index on this node)

    master_addr = os.environ["MASTER_ADDR"]
    master_port = os.environ["MASTER_PORT"]

    print("============================", flush=True)
    print(f"[Rank {rank}] DDP setup, master_addr: {master_addr}", flush=True)
    print(f"[Rank {rank}] DDP setup, master_port: {master_port}", flush=True)
    print(f"[Rank {rank}] DDP setup, rank: {rank}", flush=True)
    print(f"[Rank {rank}] DDP setup, local_rank: {local_rank}", flush=True)
    print(f"[Rank {rank}] DDP setup, world_size: {world_size}", flush=True)
    print("============================", flush=True)

    # ----- 2) Set the current GPU device for this process -----
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # ----- 3) Initialize the process group -----
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        world_size=world_size,
        rank=rank,
    )

    return rank, world_size, local_rank, device


def cleanup_distributed() -> None:
    """Cleans up distributed training using PyTorch DDP."""
    # ----- 8) Clean up (optional) -----
    dist.destroy_process_group()


def main(
        args: argparse.Namespace,
        rank: int,
        world_size: int,
        local_rank: int,
        device: torch.device
        ) -> None:
    """Main function for training a Gaussian Policy network using DDP."""
    #############################################
    # Process Inputs
    #############################################
    # Study ID
    studyIDX = args.studyIDX

    # Data Paths
    design_file = os.path.abspath(args.LSC_DESIGN_DIR + args.design_file)
    train_filelist = args.FILELIST_DIR + args.train_filelist
    validation_filelist = args.FILELIST_DIR + args.validation_filelist

    # Number of workers controls how batches of data are prefetched and,
    # possibly, pre-loaded onto GPUs. If the number of workers is large they
    # will swamp memory and jobs will fail.
    num_workers = args.num_workers

    # Epoch Parameters
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

    # Dictionary of available models.
    available_models = {
        "gaussian_policyCNN": gaussian_policyCNN
    }

    #############################################
    # Model Arguments for Dynamic Reconstruction
    #############################################
    model_args = {
        "img_size": (1, 1120, 800),
        "input_vector_size": 28,
        "output_dim": 28,
        "min_variance": 1e-6,
        "features": 12,
        "depth": 15,
        "kernel": 3,
        "img_embed_dim": 32,
        "vector_embed_dim": 32,
        "size_reduce_threshold": (16, 16),
        "vector_feature_list": (16, 64, 64, 16),
        "output_feature_list": (16, 64, 64, 16)
    }

    model = gaussian_policyCNN(**model_args)

    #############################################
    # Freeze covariance parameters
    #############################################
    for param in model.cov_mlp.parameters():
        param.requires_grad = False

    #############################################
    # Initialize Optimizer
    #############################################
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.01
    )

    #############################################
    # Initialize Loss
    #############################################
    # Use `reduction='none'` so loss on each sample in batch can be recorded.
    loss_fn = nn.MSELoss(reduction="none")

    #############################################
    # Load Model for Continuation (Rank 0 only)
    #############################################
    # Wait to move model to GPU until after the checkpoint load. Then
    # explicitly move model and optimizer state to GPU.
    if CONTINUATION:
        model, starting_epoch = tr.load_model_and_optimizer(
            checkpoint,
            optimizer,
            available_models,
            device=device,
        )

        # Freeze parameters of loaded model
        for param in model.cov_mlp.parameters():
            param.requires_grad = False

        print("Model state loaded for continuation.")
    else:
        model.to(device)
        starting_epoch = 0

    #############################################
    # Move Model to DistributedDataParallel
    #############################################
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    #############################################
    # Data Initialization (Distributed Dataloader)
    #############################################
    train_dataset = LSC_hfield_policy_DataSet(
        args.LSC_NPZ_DIR,
        filelist=train_filelist,
        design_file=design_file,
        half_image=False,
        field_list=["density_throw"]
    )
    val_dataset = LSC_hfield_policy_DataSet(
        args.LSC_NPZ_DIR,
        filelist=validation_filelist,
        design_file=design_file,
        half_image=False,
        field_list=["density_throw"]
    )

    # NOTE: For DDP the batch_size is the per-GPU batch_size!!!
    train_dataloader = tr.make_distributed_dataloader(
        train_dataset,
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        rank=rank,
        world_size=world_size,
    )
    val_dataloader = tr.make_distributed_dataloader(
        val_dataset,
        batch_size,
        shuffle=False,
        num_workers=num_workers,
        rank=rank,
        world_size=world_size,
    )
    
    #############################################
    # Training Loop (Modified for DDP)
    #############################################
    # Train Model
    print("Training Model . . .")
    starting_epoch += 1
    ending_epoch = min(starting_epoch + cycle_epochs, total_epochs + 1)

    TIME_EPOCH = True
    for epochIDX in range(starting_epoch, ending_epoch):
        train_sampler = train_dataloader.sampler
        train_sampler.set_epoch(epochIDX)

        # For timing epochs
        if TIME_EPOCH:
            # Synchronize before starting the timer
            dist.barrier()  # Ensure that all nodes sync
            torch.cuda.synchronize(device)  # Ensure GPUs on each node sync
            # Time each epoch and print to stdout
            startTime = time.time()

        # Train and Validate
        tr.train_lsc_policy_epoch(
            training_data=train_dataloader,
            validation_data=val_dataloader,
            num_train_batches=train_batches,
            num_val_batches=val_batches,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epochIDX=epochIDX,
            train_per_val=train_per_val,
            train_rcrd_filename=trn_rcrd_filename,
            val_rcrd_filename=val_rcrd_filename,
            device=device,
            rank=rank,
            world_size=world_size,
        )

        if TIME_EPOCH:
            # Synchronize before stopping the timer
            torch.cuda.synchronize(device)  # Ensure GPUs on each node sync
            dist.barrier()  # Ensure that all nodes sync
            # Time each epoch and print to stdout
            endTime = time.time()

        epoch_time = (endTime - startTime) / 60

        # Print Summary Results
        if rank == 0:
            print(f"Completed epoch {epochIDX}...", flush=True)
            print(f"Epoch time (minutes): {epoch_time:.2f}", flush=True)

    # Save model and optimizer state in hdf5
    chkpt_name_str = "study{0:03d}_modelState_epoch{1:04d}.pth"
    new_chkpt_path = os.path.join("./", chkpt_name_str.format(studyIDX, epochIDX))

    tr.save_model_and_optimizer(
        model,
        optimizer,
        epochIDX,
        new_chkpt_path,
        model_class=gaussian_policyCNN,
        model_args=model_args
    )

    if rank == 0:
        #############################################
        # Continue if Necessary
        #############################################
        FINISHED_TRAINING = epochIDX + 1 > total_epochs
        if not FINISHED_TRAINING:
            new_slurm_file = tr.continuation_setup(
                new_chkpt_path, studyIDX, last_epoch=epochIDX
            )
            os.system(f"sbatch {new_slurm_file}")


if __name__ == "__main__":
    args = parser.parse_args()

    rank, world_size, local_rank, device = setup_distributed()

    main(args, rank, world_size, local_rank, device)

    cleanup_distributed()
