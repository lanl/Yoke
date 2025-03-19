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
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
import torch
import torch.nn as nn

from yoke.models.vit.swin.bomberman import LodeRunner, Lightning_LodeRunner
from yoke.datasets.lsc_dataset import LSC_rho2rho_sequential_DataSet
import yoke.torch_training_utils as tr
from yoke.lr_schedulers import CosineWithWarmupScheduler
from yoke.helpers import cli
import yoke.scheduled_sampling


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
        patch_size=(5, 5),  # Since using half-image, halve patch size.
        embed_dim=args.embed_dim,
        emb_factor=2,
        num_heads=8,
        block_structure=tuple(args.block_structure),
        window_sizes=[(2, 2) for _ in range(4)],
        patch_merge_scales=[(2, 2) for _ in range(3)],
    )

    #############################################
    # Initialize Optimizer
    #############################################
    # Using LR scheduler so optimizer LR is fixed and small.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.anchor_lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.01,
    )

    #############################################
    # Initialize Data
    #############################################
    train_dataset = LSC_rho2rho_sequential_DataSet(
        LSC_NPZ_DIR=args.LSC_NPZ_DIR,
        file_prefix_list=train_filelist,
        max_file_checks=10,
        seq_len=args.seq_len,  # Sequence length
        half_image=True,
    )

    val_dataset = LSC_rho2rho_sequential_DataSet(
        LSC_NPZ_DIR=args.LSC_NPZ_DIR,
        file_prefix_list=validation_filelist,
        max_file_checks=10,
        seq_len=args.seq_len,  # Sequence length
        half_image=True,
    )

    #############################################
    # Training
    #############################################
    # Setup Dataloaders
    train_dataloader = tr.make_dataloader(
        train_dataset,
        args.batch_size,
        args.train_batches,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )
    val_dataloader = tr.make_dataloader(
        val_dataset,
        args.batch_size,
        args.val_batches,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )

    #############################################
    # Lightning wrap
    #############################################
    # Get start_epoch from checkpoint filename
    # Format: study{args.studyIDX:03d}_epoch={epoch:04d}_val_loss={val_loss:.4f}.ckpt
    if args.continuation:
        starting_epoch = int(
            re.search(r"epoch=(?P<epoch>\d+)_", args.checkpoint)["epoch"]
        )
        last_epoch = args.train_batches * (starting_epoch - 1)
    else:
        starting_epoch = 0
        last_epoch = -1

    in_vars = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    out_vars = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    L_loderunner = Lightning_LodeRunner(
        model,
        in_vars=in_vars,
        out_vars=out_vars,
        loss_fn=nn.MSELoss(reduction="none"),
        lr_scheduler=CosineWithWarmupScheduler,
        scheduler_params={
            "warmup_steps": args.warmup_steps,
            "anchor_lr": args.anchor_lr,
            "terminal_steps": args.terminal_steps,
            "num_cycles": args.num_cycles,
            "min_fraction": args.min_fraction,
            # "last_epoch": last_epoch,  # Lightning takes care of this automatically
        },
        scheduled_sampling_scheduler=getattr(yoke.scheduled_sampling, args.schedule)(
            initial_schedule_prob=args.initial_schedule_prob,
            decay_param=args.decay_param,
            minimum_schedule_prob=args.minimum_schedule_prob,
        ),
    )

    # Prepare Lightning trainer.
    logger = L.loggers.CSVLogger(
        save_dir="./",
        flush_logs_every_n_steps=32,
    )

    cycle_epochs = min(args.cycle_epochs, args.total_epochs - starting_epoch + 1)
    final_epoch = starting_epoch + cycle_epochs - 1
    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,
        every_n_epochs=args.TRAIN_PER_VAL,
        monitor="val_loss",
        mode="min",
        dirpath="./checkpoints",
        filename=f"study{args.studyIDX:03d}" + "_{epoch:04d}_{val_loss:.4f}",
        save_last=True,
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = f"study{args.studyIDX:03d}" + "_{epoch:04d}_{val_loss:.4f}-last"
    lr_monitor = LearningRateMonitor(logging_interval="step")
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
        log_every_n_steps=32,
        callbacks=[checkpoint_callback, lr_monitor],
    )

    # Run training using Lightning.
    if args.continuation:
        ckpt_path = args.checkpoint
    else:
        ckpt_path = None
    trainer.fit(
        L_loderunner,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=ckpt_path,
    )

    #############################################
    # Continue if Necessary
    #############################################
    # Run only in main process, otherwise we'll get NGPUs copies of the chain due
    # to the way Lightning tries to parallelize the script.
    if trainer.is_global_zero:
        FINISHED_TRAINING = final_epoch + 1 > args.total_epochs
        if not FINISHED_TRAINING:
            new_slurm_file = tr.continuation_setup(
                checkpoint_callback.last_model_path,
                args.studyIDX,
                last_epoch=final_epoch,
            )
            os.system(f"sbatch {new_slurm_file}")
