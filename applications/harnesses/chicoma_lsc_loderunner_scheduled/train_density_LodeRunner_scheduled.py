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

from yoke.models.vit.swin.bomberman import LodeRunner
from yoke.datasets.lsc_dataset import LSC_rho2rho_sequential_DataSet
import yoke.torch_training_utils as tr
from yoke.parallel_utils import LodeRunner_DataParallel
from yoke.lr_schedulers import CosineWithWarmupScheduler
from yoke.helpers import cli


#############################################
# Inputs
#############################################
descr_str = (
    "Trains LodeRunner architecture using scheduled sampling "
    "with sequential inputs and outputs. "
    "This training uses lsc240420 data for per-material density fields."
)
parser = argparse.ArgumentParser(
    prog="Initial LodeRunner Training", description=descr_str, fromfile_prefix_chars="@"
)
parser = cli.add_default_args(parser=parser)
parser = cli.add_filepath_args(parser=parser)
parser = cli.add_computing_args(parser=parser)
parser = cli.add_model_args(parser=parser)
parser = cli.add_training_args(parser=parser)
parser = cli.add_cosine_lr_scheduler_args(parser=parser)
parser = cli.add_scheduled_sampling_args(parser=parser)


#############################################
#############################################
if __name__ == "__main__":
    #############################################
    # Process Inputs
    #############################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    #############
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    #############
    # Study ID
    studyIDX = args.studyIDX

    # Data Paths
    train_filelist = os.path.join(args.FILELIST_DIR, args.train_filelist)
    validation_filelist = os.path.join(args.FILELIST_DIR, args.validation_filelist)
    test_filelist = os.path.join(args.FILELIST_DIR, args.test_filelist)

    # Model Parameters
    embed_dim = args.embed_dim
    block_structure = tuple(args.block_structure)

    # Training Parameters
    batch_size = args.batch_size

    # Scheduled Sampling Parameters
    scheduled_prob = args.initial_schedule_prob
    decay_rate = args.decay_param
    minimum_schedule_prob = args.minimum_schedule_prob

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
    model = LodeRunner(
        default_vars=[
            "density_case",
            "density_cushion",
            "density_maincharge",
            "density_outside_air",
            "density_striker",
            "density_throw",
            "Uvelocity",
            "Wvelocity",
        ],
        image_size=(1120, 400),
        patch_size=(10, 5),
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
        lr=args.anchor_lr,
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
    # We will take a scheduler step every back-prop step so the number of steps
    # is the number of previous batches.
    if starting_epoch == 0:
        last_epoch = -1
    else:
        last_epoch = train_batches * (starting_epoch - 1)
    LRsched = CosineWithWarmupScheduler(
        optimizer,
        anchor_lr=args.anchor_lr,
        terminal_steps=args.terminal_steps,
        warmup_steps=args.warmup_steps,
        num_cycles=args.num_cycles,
        min_fraction=args.min_fraction,
        last_epoch=last_epoch,
    )

    #############################################
    # Initialize Data
    #############################################
    train_dataset = LSC_rho2rho_sequential_DataSet(
        LSC_NPZ_DIR=args.LSC_NPZ_DIR,
        file_prefix_list=train_filelist,
        max_file_checks=10,
        seq_len=3,  # Sequence length
        half_image=True,
    )

    val_dataset = LSC_rho2rho_sequential_DataSet(
        LSC_NPZ_DIR=args.LSC_NPZ_DIR,
        file_prefix_list=validation_filelist,
        max_file_checks=10,
        seq_len=3,  # Sequence length
        half_image=True,
    )

    test_dataset = LSC_rho2rho_sequential_DataSet(
        LSC_NPZ_DIR=args.LSC_NPZ_DIR,
        file_prefix_list=test_filelist,
        max_file_checks=10,
        seq_len=3,  # Sequence length
        half_image=True,
    )

    print("Datasets initialized...")

    #############################################
    # Training Loop
    #############################################
    # Train Model
    print("Training Model . . .")
    starting_epoch += 1
    ending_epoch = min(starting_epoch + cycle_epochs, total_epochs + 1)

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

    for epochIDX in range(starting_epoch, ending_epoch):
        # Time each epoch and print to stdout
        startTime = time.time()

        torch.cuda.empty_cache()

        # Train an epoch with scheduled sampling
        tr.train_scheduled_loderunner_epoch(
            training_data=train_dataloader,
            validation_data=val_dataloader,
            model=model,
            optimizer=optimizer,
            LRsched=LRsched,
            loss_fn=loss_fn,
            epochIDX=epochIDX,
            train_per_val=train_per_val,
            train_rcrd_filename=trn_rcrd_filename,
            val_rcrd_filename=val_rcrd_filename,
            device=device,
            scheduled_prob=scheduled_prob,
        )

        endTime = time.time()
        epoch_time = (endTime - startTime) / 60

        # Print Summary Results
        print(
            f"Completed epoch {epochIDX} with scheduled_prob {scheduled_prob:.4f}...",
            flush=True,
        )
        print(f"Epoch time (minutes): {epoch_time:.2f}", flush=True)

        # Decay the scheduled probability
        scheduled_prob = max(scheduled_prob * decay_rate, minimum_schedule_prob)

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
