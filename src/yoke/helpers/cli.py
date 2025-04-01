"""Collection of helper functions related to yoke command line interface (CLI)."""

import argparse
import os


# Set the yoke root path relative to this file for usage in defaults below.
YOKE_PATH = os.path.join(os.path.dirname(__file__), "../../..")


def add_default_args(parser: argparse.ArgumentParser = None) -> argparse.ArgumentParser:
    """Prepare a default ArgumentParser for harnesses.

    Args:
        parser (argparse.ArgumentParser, optional): An optional arg parser.

    Returns:
        argparse.ArgumentParser: The parser with more arguments.
    """
    if parser is None:
        parser = argparse.ArgumentParser(
            prog="HARNESS START", description="Starts execution of training harness"
        )
    parser.add_argument(
        "--csv",
        action="store",
        type=str,
        default="./hyperparameters.csv",
        help="CSV file containing study hyperparameters",
    )
    parser.add_argument(
        "--studyIDX",
        action="store",
        type=int,
        default=1,
        help="Study ID number to match hyperparameters",
    )
    parser.add_argument(
        "--rundir",
        action="store",
        type=str,
        default="./runs",
        help=(
            "Directory to create study directories within. This should be a softlink to "
            "somewhere with a lot of drive space."
        ),
    )
    parser.add_argument(
        "--cpFile",
        action="store",
        type=str,
        default="./cp_files.txt",
        help=(
            "Name of text file containing local files that should be copied to the "
            "study directory."
        ),
    )

    return parser


def add_filepath_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add filepath related arguments to parser for harnesses.

    Args:
        parser (argparse.ArgumentParser): A parser that doesn't have filepath arguments.

    Returns:
        argparse.ArgumentParser: The argparse.ArgumentParser object with added arguments.
    """
    parser.add_argument(
        "--FILELIST_DIR",
        action="store",
        type=str,
        default=os.path.join(YOKE_PATH, "applications/filelists/"),
        help="Directory where filelists are located.",
    )
    parser.add_argument(
        "--LSC_DESIGN_DIR",
        action="store",
        type=str,
        default=os.path.join(YOKE_PATH, "data_examples/"),
        help="Directory in which LSC design.txt file lives.",
    )
    parser.add_argument(
        "--NC_DESIGN_DIR",
        action="store",
        type=str,
        default=os.path.join(YOKE_PATH, "data_examples/"),
        help="Directory in which NC design.txt file lives.",
    )
    parser.add_argument(
        "--design_file",
        action="store",
        type=str,
        default="design_lsc240420_SAMPLE.csv",
        help=".csv file that contains the truth values for data files",
    )
    parser.add_argument(
        "--LSC_NPZ_DIR",
        action="store",
        type=str,
        default=os.path.join(YOKE_PATH, "data_examples/lsc240420/"),
        help="Directory in which LSC *.npz files live.",
    )
    parser.add_argument(
        "--NC_NPZ_DIR",
        action="store",
        type=str,
        default=os.path.join(YOKE_PATH, "data_examples/nc231213/"),
        help="Directory in which NC *.npz files lives.",
    )
    parser.add_argument(
        "--train_filelist",
        action="store",
        type=str,
        default="lsc240420_train_sample.txt",
        help="Path to list of files to train on.",
    )
    parser.add_argument(
        "--validation_filelist",
        action="store",
        type=str,
        default="lsc240420_val_sample.txt",
        help="Path to list of files to validate on.",
    )
    parser.add_argument(
        "--test_filelist",
        action="store",
        type=str,
        default="lsc240420_test_sample.txt",
        help="Path to list of files to test on.",
    )
    return parser


def add_computing_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add computing-related (e.g., parallel processing) arguments to parser.

    Args:
        parser (argparse.ArgumentParser): An argument parser needing args.

    Returns:
        argparse.ArgumentParser: The parser with all needed args.
    """
    parser.add_argument(
        "--multigpu",
        action="store_true",
        help="Supports multiple GPUs on a single node.",
    )
    parser.add_argument(
        "--Ngpus", action="store", type=int, default=1, help="Number of GPUs per node."
    )
    parser.add_argument(
        "--Knodes", action="store", type=int, default=1, help="Number of nodes."
    )

    return parser


def add_model_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add model arguments to parser for harnesses in yoke.applications.harnesses.

    Args:
        parser (argparse.ArgumentParser): An argument parser object to add arguments to.

    Returns:
        argparse.ArgumentParser: The argparse.ArgumentParser object with added arguments.
    """
    parser.add_argument(
        "--featureList",
        action="store",
        type=int,
        nargs="+",
        default=[256, 128, 64, 32, 16],
        help="List of number of features in each T-convolution layer.",
    )
    parser.add_argument(
        "--linearFeatures",
        action="store",
        type=int,
        default=256,
        help="Number of features scalar inputs are mapped into prior to T-convs.",
    )
    parser.add_argument(
        "--block_structure",
        action="store",
        type=int,
        nargs="+",
        default=[1, 1, 3, 1],
        help="List of number of SW-MSA layers in each SWIN block.",
    )
    parser.add_argument(
        "--embed_dim",
        action="store",
        type=int,
        default=96,
        help="Initial embedding dimension for SWIN-Unet.",
    )

    return parser


def add_training_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add training arguments to parser for harnesses in yoke.applications.harnesses.

    Args:
        parser (argparse.ArgumentParser): An argument parser to add arguments to.

    Returns:
        argparse.ArgumentParser: The argparse.ArgumentParser object with added arguments.
    """
    parser.add_argument(
        "--batch_size", action="store", type=int, default=64, help="Batch size"
    )
    parser.add_argument(
        "--seq_len",
        action="store",
        type=int,
        default=3,
        help="Length of predicted sequence.",
    )
    parser.add_argument(
        "--total_epochs",
        action="store",
        type=int,
        default=10,
        help="Total training epochs",
    )
    parser.add_argument(
        "--cycle_epochs",
        action="store",
        type=int,
        default=5,
        help=(
            "Number of epochs between saving the model and re-queueing "
            "training process; must be able to be completed in the "
            "set wall time"
        ),
    )
    parser.add_argument(
        "--train_batches",
        action="store",
        type=int,
        default=250,
        help="Number of batches to train on in a given epoch",
    )
    parser.add_argument(
        "--val_batches",
        action="store",
        type=int,
        default=25,
        help="Number of batches to validate on in a given epoch",
    )
    parser.add_argument(
        "--TRAIN_PER_VAL",
        action="store",
        type=int,
        default=10,
        help="Number of training epochs between each validation epoch",
    )
    parser.add_argument(
        "--trn_rcrd_filename",
        action="store",
        type=str,
        default="./default_training.csv",
        help="Filename for text file of training loss and metrics on each batch",
    )
    parser.add_argument(
        "--val_rcrd_filename",
        action="store",
        type=str,
        default="./default_validation.csv",
        help="Filename for text file of validation loss and metrics on each batch",
    )
    parser.add_argument(
        "--continuation",
        action="store_true",
        help="Indicates if training is being continued or restarted",
    )
    parser.add_argument(
        "--checkpoint",
        action="store",
        type=str,
        default=None,
        help="Path to checkpoint to continue training from",
    )
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="Freeze model backbone while training.",
    )
    parser.add_argument(
        "--num_workers",
        action="store",
        type=int,
        default=4,
        help=(
            "Number of processes simultaneously loading batches of data. "
            "NOTE: If set too big workers will swamp memory!!"
        ),
    )
    parser.add_argument(
        "--prefetch_factor",
        action="store",
        type=int,
        default=2,
        help=(
            "Number of batches each worker preloads ahead of time. "
            "NOTE: If set too big preload will swamp memory!!"
        ),
    )

    return parser


def add_step_lr_scheduler_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Add StepLR arguments to parser for harnesses in yoke.applications.harnesses.

    Args:
        parser (argparse.ArgumentParser): An argument parser to add arguments to.

    Returns:
        argparse.ArgumentParser: The argparse.ArgumentParser object with added arguments.
    """
    parser.add_argument(
        "--init_learnrate",
        action="store",
        type=float,
        default=1e-3,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--LRepoch_per_step",
        action="store",
        type=float,
        default=10,
        help="Number of epochs per LR reduction.",
    )
    parser.add_argument(
        "--LRdecay", action="store", type=float, default=0.5, help="LR decay factor."
    )

    return parser


def add_cosine_lr_scheduler_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Add arguments for the cosine with warmup learning rate scheduler.

    Args:
        parser (argparse.ArgumentParser): An argument parser to add arguments to.

    Returns:
        argparse.ArgumentParser: The argparse.ArgumentParser object with added arguments.
    """
    parser.add_argument(
        "--anchor_lr",
        action="store",
        type=float,
        default=1e-4,
        help="Learning rate at the peak of cosine scheduler.",
    )
    parser.add_argument(
        "--num_cycles",
        action="store",
        type=float,
        default=0.5,
        help="Number of cycles of cosine LR schedule after initial warmup phase.",
    )
    parser.add_argument(
        "--min_fraction",
        action="store",
        type=float,
        default=0.5,
        help="Fraction of anchor-LR at cosine trough.",
    )
    parser.add_argument(
        "--terminal_steps",
        action="store",
        type=int,
        default=1000,
        help="Number of steps for cosine to go through specified cycles.",
    )
    parser.add_argument(
        "--warmup_steps",
        action="store",
        type=int,
        default=500,
        help="Number of steps for scheduler to reach anchor-LR.",
    )

    return parser


def add_scheduled_sampling_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Add scheduled sampling arguments to parser for harnesses.

    Args:
        parser (argparse.ArgumentParser): An argument parser to add arguments to.

    Returns:
        argparse.ArgumentParser: The argparse.ArgumentParser object with added arguments.
    """
    parser.add_argument(
        "--schedule",
        action="store",
        type=str,
        default="inverse_sigmoid",
        help=(
            "Name of a function in src.yoke.scheduled_sampling "
            "defining scheduled sampling schedule."
        ),
    )
    parser.add_argument(
        "--initial_schedule_prob",
        action="store",
        type=float,
        default=1.0,  # Initial probability of using ground truth
        help="Initial probability of using ground truth for scheduled sampling.",
    )
    parser.add_argument(
        "--decay_param",
        action="store",
        type=float,
        default=100.0,  # Decay parameter for scheduled_prob
        help="Parameter defining decay of scheduled sampling schedule.",
    )
    parser.add_argument(
        "--minimum_schedule_prob",
        action="store",
        type=float,
        default=0.0,  # Minimum probability of using ground truth
        help="Minimum scheduled-sampling probability.",
    )

    return parser


def add_ch_subsampling_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Add channel subsampling arguments to parser for harnesses.

    Args:
        parser (argparse.ArgumentParser): An argument parser to add arguments to.

    Returns:
        argparse.ArgumentParser: The argparse.ArgumentParser object with added arguments.
    """
    parser.add_argument(
        "--channel_map_size",
        action="store",
        type=int,
        default=3,
        help="Channel subsampling map size",
    )

    return parser
