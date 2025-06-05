"""Training for Lightning-wrapped LodeRunner on LSC material densities.

This version of training uses only the lsc240420 data with only per-material
density along with the velocity field. A single timestep is input, a single
timestep is predicted. The number of input variables is fixed throughout
training.

`lightning` is used to train a LightningModule wrapper for LodeRunner to allow
multi-node, multi-GPU, distributed data-parallel training.

"""

#############################################
# Packages
#############################################
import argparse
import os
import re

import lightning.pytorch as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
import torch
import torch.nn as nn
import numpy as np

from yoke.models.vit.swin.bomberman import LodeRunner, Lightning_LodeRunner
from yoke.datasets.lsc_dataset import LSCDataModule
from yoke.datasets.transforms import ResizePadCrop
import yoke.torch_training_utils as tr
from yoke.lr_schedulers import CosineWithWarmupScheduler
from yoke.helpers import cli
import yoke.scheduled_sampling
from yoke.losses.masked_loss import CroppedLoss2D


#############################################
# Inputs
#############################################
descr_str = (
    "Trains lightning-wrapped LodeRunner on multi-timestep input and output of the "
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
parser = cli.add_cosine_lr_scheduler_args(parser=parser)
parser = cli.add_scheduled_sampling_args(parser=parser)

# Change some default filepaths.
parser.set_defaults(
    train_filelist="lsc240420_prefixes_train_80pct.txt",
    validation_filelist="lsc240420_prefixes_validation_10pct.txt",
    test_filelist="lsc240420_prefixes_test_10pct.txt",
)


#############################################
#############################################
if __name__ == "__main__":
    # Set precision for tensor core speedup potential.
    torch.set_float32_matmul_precision("medium")

    #############################################
    # Process Inputs
    #############################################
    args = parser.parse_args()

    # Data Paths
    train_filelist = os.path.join(args.FILELIST_DIR, args.train_filelist)
    validation_filelist = os.path.join(args.FILELIST_DIR, args.validation_filelist)
    test_filelist = os.path.join(args.FILELIST_DIR, args.test_filelist)

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
    image_size = (
        args.image_size if args.scaled_image_size is None else args.scaled_image_size
    )
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
        image_size=image_size,
        patch_size=(5, 5),
        embed_dim=args.embed_dim,
        emb_factor=2,
        num_heads=8,
        block_structure=tuple(args.block_structure),
        window_sizes=[(2, 2) for _ in range(4)],
        patch_merge_scales=[(2, 2) for _ in range(3)],
    )

    #############################################
    # Initialize Data
    #############################################
    transform = ResizePadCrop(
        interp_kwargs={"scale_factor": args.scale_factor},
        scaled_image_size=args.scaled_image_size,
        pad_position=("bottom", "right"),
    )
    ds_params = {
        "LSC_NPZ_DIR": args.LSC_NPZ_DIR,
        "max_file_checks": 10,
        "seq_len": args.seq_len,
        "timeIDX_offset": args.timeIDX_offset,
        "half_image": True,
        "transform": transform,
    }
    dl_params = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor,
    }
    lsc_datamodule = LSCDataModule(
        ds_name="LSC_rho2rho_sequential_DataSet",
        ds_params_train=ds_params | {"file_prefix_list": train_filelist},
        ds_params_val=ds_params | {"file_prefix_list": validation_filelist},
        dl_params_train=dl_params | {"shuffle": True, "persistent_workers": True},
        dl_params_val=dl_params | {"shuffle": False, "persistent_workers": True},
    )

    #############################################
    # Lightning wrap
    #############################################
    # Define a cropped loss function (used to ignore padding on rescaled images).
    loss_mask = torch.zeros(args.scaled_image_size, dtype=torch.float)
    scaled_image_size = np.array(args.scaled_image_size)
    valid_im_size = np.floor(args.scale_factor * np.array(args.image_size)).astype(int)
    loss = CroppedLoss2D(
        loss_fxn=nn.MSELoss(reduction="none"),
        crop=(
            0,
            0,
            min(valid_im_size[0], args.scaled_image_size[0]),
            min(valid_im_size[1], args.scaled_image_size[1]),
        ),  # corresponds to pad_position=("bottom", "right") in ResizePadCrop
    )

    # Prepare the Lightning module.
    lm_kwargs = {
        "model": model,
        "in_vars": torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
        "out_vars": torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
        "loss_fn": loss,
        "lr_scheduler": CosineWithWarmupScheduler,
        "scheduler_params": {
            "warmup_steps": args.warmup_steps,
            "anchor_lr": args.anchor_lr,
            "terminal_steps": args.terminal_steps,
            "num_cycles": args.num_cycles,
            "min_fraction": args.min_fraction,
        },
        "scheduled_sampling_scheduler": getattr(yoke.scheduled_sampling, args.schedule)(
            initial_schedule_prob=args.initial_schedule_prob,
            decay_param=args.decay_param,
            minimum_schedule_prob=args.minimum_schedule_prob,
        ),
    }
    if args.continuation or (args.checkpoint is None) or args.only_load_backbone:
        L_loderunner = Lightning_LodeRunner(**lm_kwargs)
    else:
        # This condition is used to load pretrained weights without continuing training.
        L_loderunner = Lightning_LodeRunner.load_from_checkpoint(
            checkpoint_path=args.checkpoint,
            strict=False,
            **lm_kwargs,
        )

    # Load U-Net backbone if needed.
    if args.only_load_backbone:
        ckpt = torch.load(args.checkpoint, map_location=torch.device("cpu"))
        unet_weights = {
            k.replace("model.unet.", ""): v
            for k, v in ckpt["state_dict"].items()
            if k.startswith("model.unet.")
        }
        L_loderunner.model.unet.load_state_dict(unet_weights)

    # Freeze the U-Net backbone.
    if args.freeze_backbone:
        tr.freeze_torch_params(L_loderunner.model.unet)

    # Prepare Lightning logger.
    logger = TensorBoardLogger(save_dir="./")

    # Determine starting and final epochs for this round of training.
    # Format: study{args.studyIDX:03d}_epoch={epoch:04d}_val_loss={val_loss:.4f}.ckpt
    if args.continuation:
        starting_epoch = (
            int(re.search(r"epoch=(?P<epoch>\d+)_", args.checkpoint)["epoch"]) + 1
        )
    else:
        starting_epoch = 0

    # Prepare trainer.
    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,
        every_n_epochs=args.TRAIN_PER_VAL,
        monitor="val_loss",
        mode="min",
        dirpath="./checkpoints",
        filename=f"study{args.studyIDX:03d}" + "_{epoch:04d}_{val_loss:.4f}",
        save_last=True,
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = (
        f"study{args.studyIDX:03d}" + "_{epoch:04d}_{val_loss:.4f}-last"
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    final_epoch = min(starting_epoch + args.cycle_epochs, args.total_epochs) - 1
    trainer = L.Trainer(
        max_epochs=final_epoch + 1,
        limit_train_batches=args.train_batches,
        check_val_every_n_epoch=args.TRAIN_PER_VAL,
        limit_val_batches=args.val_batches,
        accelerator="gpu",
        devices=args.Ngpus,  # Number of GPUs per node
        num_nodes=args.Knodes,
        strategy="ddp",
        enable_progress_bar=True,
        logger=logger,
        log_every_n_steps=min(args.train_batches, args.val_batches),
        callbacks=[checkpoint_callback, lr_monitor],
    )

    # Run training using Lightning.
    if args.continuation:
        ckpt_path = args.checkpoint
    else:
        ckpt_path = None
    trainer.fit(
        L_loderunner,
        datamodule=lsc_datamodule,
        ckpt_path=ckpt_path,
    )

    #############################################
    # Continue if Necessary
    #############################################
    # Run only in main process, otherwise we'll get NGPUs copies of the chain due
    # to the way Lightning tries to parallelize the script.
    if trainer.is_global_zero:
        FINISHED_TRAINING = (final_epoch + 1) >= args.total_epochs
        if not FINISHED_TRAINING:
            new_slurm_file = tr.continuation_setup(
                checkpoint_callback.last_model_path,
                args.studyIDX,
                last_epoch=final_epoch + 1,
            )
            os.system(f"sbatch {new_slurm_file}")
