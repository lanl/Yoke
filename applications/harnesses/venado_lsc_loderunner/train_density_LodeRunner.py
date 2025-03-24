"""Initial training setup for LodeRunner on LSC material densities.

This version of training uses only the lsc240420 data with only per-material
density along with the velocity field. A single timestep is input, a single
timestep is predicted. The number of input variables is fixed throughout
training.

"""

#############################################
# Packages
#############################################
import os
import time
import argparse
import torch
import torch.nn as nn
import random
import logging
import numpy as np

from yoke.models.vit.swin.bomberman import LodeRunner
from yoke.datasets.lsc_dataset import LSC_rho2rho_temporal_DataSet
import yoke.torch_training_utils as tr
from yoke.parallel_utils import LodeRunner_DataParallel
from yoke.helpers import cli
import yoke.helpers.logger as yl

#############################################
# Inputs
#############################################
descr_str = (
    "Trains LodeRunner architecture on single-timstep input and output of the "
    "lsc240420 per-material density fields."
)
parser = argparse.ArgumentParser(
    prog="Initial LodeRunner Training", description=descr_str, fromfile_prefix_chars="@"
)
parser = cli.add_default_args(parser=parser)
parser = cli.add_filepath_args(parser=parser)
parser = cli.add_computing_args(parser=parser)
parser = cli.add_model_args(parser=parser)
parser = cli.add_training_args(parser=parser)
parser = cli.add_step_lr_scheduler_args(parser=parser)

# Change some default filepaths.
parser.set_defaults(
    train_filelist="lsc240420_prefixes_train_80pct.txt",
    validation_filelist="lsc240420_prefixes_validation_10pct.txt",
    test_filelist="lsc240420_prefixes_test_10pct.txt",
)

#############################################
# Channel Subset Study Param
#############################################
parser.add_argument(
    "--channel_map_size",
    action="store",
    type=int,
    default=0,
    help="Index into the list of tuple of input and output channel subsets",
)


############################################
# Select n channles randomly for an epoch
############################################
def rand_channel_map(
    max_number_channels: int,
    num_subchannels: int,
    seed: int = None,
    _seed_set: list[bool] = [False],
) -> list:
    """Choose list of subsampled channels."""
    if num_subchannels > max_number_channels:
        raise ValueError("Subsampled channels cannot be greater than maximum channels.")

    if seed is not None and not _seed_set[0]:
        random.seed(seed)
        _seed_set[0] = True  # Mark the seed as set

    return sorted(random.sample(range(0, max_number_channels), num_subchannels))


if __name__ == "__main__":
    #############################################
    # Process Inputs
    #############################################
    yl.configure_logger("yoke_logger", level=logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Study ID
    studyIDX = args.studyIDX

    # Data Paths
    train_filelist = args.FILELIST_DIR + args.train_filelist
    validation_filelist = args.FILELIST_DIR + args.validation_filelist
    test_filelist = args.FILELIST_DIR + args.test_filelist

    # Model Parameters
    embed_dim = args.embed_dim
    block_structure = tuple(args.block_structure)

    # Training Parameters
    initial_learningrate = args.init_learnrate
    LRepoch_per_step = args.LRepoch_per_step
    LRdecay = args.LRdecay
    batch_size = args.batch_size

    # Number of workers controls how batches of data are prefetched and,
    # possibly, pre-loaded onto GPUs. If the number of workers is large they
    # will swamp memory and jobs will fail.
    num_workers = args.num_workers
    prefetch_factor = args.prefetch_factor

    # Epoch Parameters
    total_epochs = args.total_epochs
    cycle_epochs = args.cycle_epochs
    train_batches = args.train_batches
    val_batches = args.val_batches
    train_per_val = args.TRAIN_PER_VAL
    trn_rcrd_filename = args.trn_rcrd_filename
    val_rcrd_filename = args.val_rcrd_filename
    CONTINUATION = args.continuation
    START = not CONTINUATION
    checkpoint = args.checkpoint

    #############################################
    # Check Devices
    #############################################
    print("\n")
    print("Slurm & Device Information")
    print("=========================================")
    print("Slurm Job ID:", os.environ["SLURM_JOB_ID"])
    print("Pytorch Cuda Available:", torch.cuda.is_available())
    print("GPU ID:", os.environ["SLURM_JOB_GPUS"])
    print("Number of System CPUs:", os.cpu_count())
    print("Number of CPUs per GPU:", os.environ["SLURM_JOB_CPUS_PER_NODE"])

    print("\n")
    print("Model Training Information")
    print("=========================================")

    #############################################
    # Initialize Model
    #############################################
    hydro_fields = [
        "density_case",
        "density_cushion",
        "density_maincharge",
        "density_outside_air",
        "density_striker",
        "density_throw",
        "Uvelocity",
        "Wvelocity",
    ]

    model = LodeRunner(
        default_vars=hydro_fields,
        image_size=(1120, 800),
        patch_size=(10, 10),
        embed_dim=embed_dim,
        emb_factor=2,
        num_heads=8,
        block_structure=block_structure,
        window_sizes=[
            (8, 8),
            (8, 8),
            (4, 4),
            (2, 2),
        ],
        patch_merge_scales=[
            (2, 2),
            (2, 2),
            (2, 2),
        ],
    )

    print("Lode Runner parameters:", tr.count_torch_params(model, trainable=True))
    # Wait to move model to GPU until after the checkpoint load. Then
    # explicitly move model and optimizer state to GPU.

    #############################################
    # Initialize Optimizer
    #############################################
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=initial_learningrate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.01,
    )

    #############################################
    # Initialize Loss
    #############################################
    # Use `reduction='none'` so loss on each sample in batch can be recorded.
    loss_fn = nn.MSELoss(reduction="none")

    print("Model initialized.")

    #############################################
    # Load Model for Continuation
    #############################################
    if CONTINUATION:
        starting_epoch = tr.load_model_and_optimizer_hdf5(model, optimizer, checkpoint)
        print("Model state loaded for continuation.")
    else:
        starting_epoch = 0

    #############################################
    # Move model and optimizer state to GPU
    #############################################
    if args.multigpu:
        model = LodeRunner_DataParallel(model)

    model.to(device)

    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    #############################################
    # Setup LR scheduler
    #############################################
    stepLRsched = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=LRepoch_per_step,
        gamma=LRdecay,
        last_epoch=starting_epoch - 1,
    )

    #############################################
    # Training Loop
    #############################################
    # Train Model
    print("Training Model . . .")
    starting_epoch += 1
    ending_epoch = min(starting_epoch + cycle_epochs, total_epochs + 1)

    # For reproducibility of channel map per epoch
    SEED = 42

    max_channels = len(hydro_fields)
    yl.logger.info(f"Max Channel  : {max_channels}")

    channel_map_size = args.channel_map_size
    yl.logger.info(f"channel_map_size = {channel_map_size}")

    # Change hydrofields to array to enable slicing with channel map
    hydro_fields = np.array(hydro_fields)
    for epochIDX in range(starting_epoch, ending_epoch):
        # Randomly select 'channel_map_size' number of channels from for the epoch
        channel_map = rand_channel_map(max_channels, channel_map_size, SEED)

        log_str = (
            f"Epoch {epochIDX:04d}, "
            f"Nchannels:{channel_map_size:03d}, "
            f"Channel Map:{channel_map}"
        )
        yl.logger.info(log_str)

        #############################################
        # Initialize Data
        # For varying channels subset per epoch, the
        # data must be initialized for each epoch.
        #############################################
        train_dataset = LSC_rho2rho_temporal_DataSet(
            args.LSC_NPZ_DIR,
            file_prefix_list=train_filelist,
            max_timeIDX_offset=2,  # This could be a variable.
            max_file_checks=10,
            hydro_fields=hydro_fields[channel_map],
        )
        val_dataset = LSC_rho2rho_temporal_DataSet(
            args.LSC_NPZ_DIR,
            file_prefix_list=validation_filelist,
            max_timeIDX_offset=2,  # This could be a variable.
            max_file_checks=10,
            hydro_fields=hydro_fields[channel_map],
        )
        test_dataset = LSC_rho2rho_temporal_DataSet(
            args.LSC_NPZ_DIR,
            file_prefix_list=test_filelist,
            max_timeIDX_offset=2,  # This could be a variable.
            max_file_checks=10,
            hydro_fields=hydro_fields[channel_map],
        )

        print("Datasets initialized...")

        # Setup Dataloaders
        train_dataloader = tr.make_dataloader(
            train_dataset,
            batch_size,
            train_batches,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )
        val_dataloader = tr.make_dataloader(
            val_dataset,
            batch_size,
            val_batches,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )
        print("DataLoaders initialized...")

        # Time each epoch and print to stdout
        startTime = time.time()

        # Train an Epoch
        tr.train_simple_loderunner_epoch(
            channel_map,
            training_data=train_dataloader,
            validation_data=val_dataloader,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epochIDX=epochIDX,
            train_per_val=train_per_val,
            train_rcrd_filename=trn_rcrd_filename,
            val_rcrd_filename=val_rcrd_filename,
            device=device,
            verbose=False,
        )

        # Increment LR scheduler
        stepLRsched.step()

        endTime = time.time()
        epoch_time = (endTime - startTime) / 60

        # Print Summary Results
        print("Completed epoch " + str(epochIDX) + "...", flush=True)
        print("Epoch time (minutes):", epoch_time, flush=True)

        # Clear GPU memory after each epoch
        torch.cuda.empty_cache()

    # Save Model Checkpoint
    print("Saving model checkpoint at end of epoch " + str(epochIDX) + ". . .")

    # Move the model back to CPU prior to saving to increase portability
    model.to("cpu")
    # Move optimizer state back to CPU
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to("cpu")

    # Save model and optimizer state in hdf5
    h5_name_str = "study{0:03d}_modelState_epoch{1:04d}.hdf5"
    new_h5_path = os.path.join("./", h5_name_str.format(studyIDX, epochIDX))
    tr.save_model_and_optimizer_hdf5(
        model, optimizer, epochIDX, new_h5_path, compiled=False
    )

    #############################################
    # Continue if Necessary
    #############################################
    FINISHED_TRAINING = epochIDX + 1 > total_epochs
    if not FINISHED_TRAINING:
        new_slurm_file = tr.continuation_setup(
            new_h5_path, studyIDX, last_epoch=epochIDX
        )
        os.system(f"sbatch {new_slurm_file}")

    ###########################################################################
    # For array prediction, especially large array prediction, the network is
    # not evaluated on the test set after training. This is performed using
    # the *evaluation* module as a separate post-analysis step.
    ###########################################################################
