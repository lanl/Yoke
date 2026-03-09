"""Evaluate trained model on test set.

Model is evaluated on `cycle_epoch` number of epochs with `test_batches` number of
batches each of a set `batch_size`.

"""

import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from yoke.models.vit.swin.bomberman import LodeRunner
from yoke.datasets.lsc_dataset import LSC_rho2rho_temporal_DataSet
from yoke.utils.checkpointing import load_model_and_optimizer
from yoke.utils.training.epoch.loderunner import eval_loderunner_epoch
from yoke.helpers import cli


descr_str = (
    "Single-GPU evaluation for a saved Yoke LodeRunner checkpoint on real dataset "
    "batches. Mirrors train_LodeRunner_ddp.py structure but without DDP."
)
parser = argparse.ArgumentParser(
    prog="LodeRunner Evaluation", description=descr_str, fromfile_prefix_chars="@"
)

# Reuse the same CLI arg groups as training so you can pass the same @argfiles.
parser = cli.add_default_args(parser=parser)
parser = cli.add_filepath_args(parser=parser)
parser = cli.add_computing_args(parser=parser)
parser = cli.add_training_args(parser=parser)

# Keep the same default filelists as train_LodeRunner_ddp.py
parser.set_defaults(
    train_filelist="lsc240420_prefixes_train_80pct.txt",
    validation_filelist="lsc240420_prefixes_validation_10pct.txt",
    test_filelist="lsc240420_prefixes_test_10pct.txt",
)

# Eval-specific args
parser.add_argument(
    "--pretrained_model",
    type=str,
    default="./study005_modelState_epoch0100.pth",
    help=(
        "Path to .pth created by yoke.utils.checkpointing.save_model_and_optimizer.",
    )
)

# Number of channels
parser.add_argument(
    "--number_channels",
    type=int,
    default=8,
    help=(
        "Number of channels in model for channel_map.",
    )
)
def main(args: argparse.Namespace) -> None:
    """Main evaluation function."""
    #############################################
    # Process Inputs
    #############################################
    # Device (single GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    filelist_dir = args.FILELIST_DIR
    filelist_path = os.path.join(filelist_dir, args.test_filelist)

    # Dataloader params
    batch_size = args.batch_size
    num_workers = args.num_workers
    test_batches = args.test_batches
    batch_size = args.batch_size
    cycle_epochs = args.cycle_epochs
    test_rcrd_filename = args.test_rcrd_filename

    #############################################
    # Load Model Checkpoint
    #############################################
    available_models = {"LodeRunner": LodeRunner}

    # NOTE: optimizer args are required by load_model_and_optimizer, even for eval.
    model, _optimizer, starting_epoch = load_model_and_optimizer(
        args.pretrained_model,
        optimizer_class=torch.optim.AdamW,
        optimizer_kwargs={
            "lr": 1e-6,
            "betas": (0.9, 0.999),
            "eps": 1e-08,
            "weight_decay": 0.01,
        },
        available_models=available_models,
        device=device,
    )
    starting_epoch = 0
    model.to(device)
    model.eval()

    # load_and_eval_YokePth.py prints these; keep similar behavior here
    print(f"Loaded checkpoint: {args.pretrained_model}", flush=True)
    print(f"Checkpoint starting_epoch: {starting_epoch}", flush=True)
    if hasattr(model, "default_vars"):
        print("Default LodeRunner fields:", model.default_vars, flush=True)
    if hasattr(model, "image_size"):
        print("LodeRunner image size:", model.image_size, flush=True)

    #############################################
    # Dataset / Dataloader (non-distributed)
    #############################################
    testing_dataset = LSC_rho2rho_temporal_DataSet(
        args.LSC_NPZ_DIR,
        file_prefix_list=filelist_path,
        max_timeIDX_offset=2,
        max_file_checks=10,
        half_image=True,
    )

    test_dataloader = DataLoader(
        dataset=testing_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    #############################################
    # Loss + Evaluation Loop
    #############################################
    # Match train_LodeRunner_ddp.py: use per-element MSE so we can reduce ourselves
    loss_fn = nn.MSELoss(reduction="none")

    # This is the natural channel map for the model trained in train_LodeRunner_ddp.py
    # (8 default vars, indices 0..7).
    channel_map = list(range(args.number_channels))

    #############################################
    # Testing Loop
    #############################################
    print("Testining Model . . .")
    starting_epoch += 1
    ending_epoch = starting_epoch + cycle_epochs

    for epochIDX in range(starting_epoch, ending_epoch):
        # Time each epoch and print to stdout
        startTime = time.time()

        # Testing epoch
        eval_loderunner_epoch(
            testing_data=test_dataloader,
            num_test_batches=test_batches,
            model=model,
            channel_map=channel_map,
            loss_fn=loss_fn,
            epochIDX=epochIDX,
            test_rcrd_filename=test_rcrd_filename,
            device=device,
        )

        # Time each epoch and print to stdout
        endTime = time.time()

        epoch_time = (endTime - startTime) / 60

        # Print Summary Results
        print(f"Completed epoch {epochIDX}...", flush=True)
        print(f"Epoch time (minutes): {epoch_time:.2f}", flush=True)


if __name__ == "__main__":
    """Parse arguments and run main evaluation function."""

    args = parser.parse_args()

    main(args)
