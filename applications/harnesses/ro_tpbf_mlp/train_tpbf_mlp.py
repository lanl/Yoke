"""Single-process CPU harness: SwiGLU MLP on the TPBF melt-pool table.

Thin wrapper around :class:`yoke.harnesses.trainer.HarnessTrainer` in CPU mode
(``device_type="cpu"``), written for the CPU-only rocinante cluster. One study
is one k-fold split: it trains on four folds of the melted rows of
``tpbf_training_set_261105.csv`` and validates on the held-out fold.

- Data: :class:`yoke.datasets.tabular_csv.TabularCSV_DataSet`. Inputs and
  targets are z-scored with the training fold's statistics (``S_wpct`` is
  ``log10``-ed first), and folds are stratified by ``stratum``.
- Model: :class:`yoke.models.mlp.SwiGLU_mlp`.
- Loss: :class:`yoke.losses.lp_loss.VectorLpLoss`, the per-sample L2 norm of
  the z-scored error over the target vector. Recorded losses are in z units.
- Epoch: :func:`yoke.utils.training.epoch.array_output.train_DDP_array_epoch`
  (array contract, ``(inpt, truth)``). It steps the LR scheduler every batch,
  so a scheduler is always built.

Column lists are colon-separated (``--input_cols power_W:sigma_m``) so that a
whole input set fits in one study-CSV cell and can be swept.

The validation loader drops the last incomplete batch (``drop_last=True``),
and it does not shuffle, so the same 0-3 holdout rows are left out of the
recorded validation loss every epoch. ``summarize_kfold.py`` scores every
holdout row from the final checkpoint.
"""

import argparse
import functools

from yoke.datasets.tabular_csv import TabularCSV_DataSet
from yoke.harnesses.trainer import HarnessTrainer
from yoke.helpers import cli
from yoke.losses.lp_loss import VectorLpLoss
from yoke.lr_schedulers import CosineWithWarmupScheduler
from yoke.models.mlp import SwiGLU_mlp
from yoke.utils.builders import build_adamw, build_from_checkpoint
from yoke.utils.training.epoch.array_output import train_DDP_array_epoch

#############################################
# Inputs
#############################################
descr_str = (
    "Trains a SwiGLU MLP on one k-fold split of a row-per-simulation CSV, as a "
    "single CPU process."
)
parser = argparse.ArgumentParser(
    prog="TPBF melt-pool MLP Training (CPU)",
    description=descr_str,
    fromfile_prefix_chars="@",
)
parser = cli.add_default_args(parser=parser)
parser = cli.add_computing_args(parser=parser)
parser = cli.add_training_args(parser=parser)
parser = cli.add_cosine_lr_scheduler_args(parser=parser)


def colon_list(value: str) -> list[str]:
    """Split a colon-separated column list, dropping empty entries.

    Args:
        value (str): For example ``"power_W:scan_speed_mps"``.

    Returns:
        list[str]: The column names.
    """
    return [v for v in value.split(":") if v]


def log_col_list(value: str) -> list[str]:
    """Like :func:`colon_list`, but ``"none"`` (any case) means no columns.

    A study-CSV cell can't be empty (it would render as ``nan``), so studies
    without a log-transformed column, e.g. those without ``S_wpct``, pass
    ``none``.

    Args:
        value (str): For example ``"S_wpct"`` or ``"none"``.

    Returns:
        list[str]: The column names, possibly empty.
    """
    return [] if value.strip().lower() == "none" else colon_list(value)


# Data and split.
parser.add_argument(
    "--table_csv",
    type=str,
    default="./tpbf_training_set_261105.csv",
    help="Row-per-simulation CSV (copied into each study dir).",
)
parser.add_argument(
    "--input_cols",
    type=colon_list,
    default=colon_list("power_W:scan_speed_mps:sigma_m:S_wpct"),
    help="Colon-separated input columns.",
)
parser.add_argument(
    "--target_cols",
    type=colon_list,
    default=colon_list("depth_um:width_um:fwhm_um:length_um"),
    help="Colon-separated target columns.",
)
parser.add_argument(
    "--log_cols",
    type=log_col_list,
    default=colon_list("S_wpct"),
    help="Colon-separated columns to log10 before z-scoring ('none' for none).",
)
parser.add_argument(
    "--require_true",
    type=colon_list,
    default=colon_list("melted"),
    help="Colon-separated boolean columns that must be True to keep a row.",
)
parser.add_argument(
    "--stratify_col",
    type=str,
    default="stratum",
    help="Column used to stratify the folds ('' for none).",
)
parser.add_argument("--n_folds", type=int, default=5, help="Number of folds.")
parser.add_argument("--fold", type=int, default=0, help="Held-out fold index.")
parser.add_argument(
    "--split_seed", type=int, default=0, help="Seed of the fold shuffle."
)


def colon_ints(value: str) -> list[int]:
    """Split a colon-separated list of integers (``"32:32"`` -> ``[32, 32]``).

    Args:
        value (str): Integers separated by colons; a single integer is allowed.

    Returns:
        list[int]: The integers.
    """
    return [int(v) for v in value.split(":") if v]


# Model. Widths may be given as separate tokens (64 64 64) or as one
# colon-separated token (32:32), which fits in a single study-CSV cell; the
# per-token lists are flattened in make_model_args.
parser.add_argument(
    "--hidden_dims",
    type=colon_ints,
    nargs="+",
    default=[[64, 64, 64]],
    help="Width of each hidden layer, e.g. '64 64 64' or '32:32'.",
)
parser.add_argument(
    "--no_residual", action="store_true", help="Disable skip connections."
)
parser.add_argument(
    "--no_layer_norm", action="store_true", help="Disable the pre-LayerNorm."
)
parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate.")

# Optimizer.
parser.add_argument(
    "--weight_decay", type=float, default=0.01, help="AdamW weight decay."
)


def make_model_args(args: argparse.Namespace) -> dict:
    """Build the ``SwiGLU_mlp`` constructor arguments.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        dict: Keyword arguments for :class:`SwiGLU_mlp`, saved in checkpoints.
    """
    return {
        "input_dim": len(args.input_cols),
        "output_dim": len(args.target_cols),
        "hidden_dims": [w for token in args.hidden_dims for w in token],
        "residual": not args.no_residual,
        "layer_norm": not args.no_layer_norm,
        "dropout": args.dropout,
        # Never layer-normalize the 4-feature input: that discards each row's
        # mean and scale across features (bug found 2026-11-18).
        "input_layer_norm": False,
    }


def build_split(args: argparse.Namespace, split: str) -> TabularCSV_DataSet:
    """Build the ``"train"`` or ``"holdout"`` dataset for this study's fold.

    Both splits are built from identical arguments, so they are exact
    complements and share the training-fold normalization.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        split (str): ``"train"`` or ``"holdout"``.

    Returns:
        TabularCSV_DataSet: The dataset for ``split``.
    """
    return TabularCSV_DataSet(
        args.table_csv,
        args.input_cols,
        args.target_cols,
        split=split,
        n_folds=args.n_folds,
        fold=args.fold,
        split_seed=args.split_seed,
        require_true=args.require_true,
        stratify_col=args.stratify_col or None,
        log_cols=args.log_cols,
    )


def build_dataset(
    args: argparse.Namespace,
) -> tuple[TabularCSV_DataSet, TabularCSV_DataSet]:
    """Build the (train, validation) datasets; validation is the held-out fold.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        tuple: ``(train_dataset, holdout_dataset)``.
    """
    return build_split(args, "train"), build_split(args, "holdout")


def optimizer_kwargs(args: argparse.Namespace) -> dict:
    """AdamW settings, shared by fresh runs and continuation reloads.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        dict: Keyword arguments for :class:`torch.optim.AdamW`.
    """
    return {
        "lr": args.anchor_lr,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": args.weight_decay,
    }


def build_optimizer(model: object, args: argparse.Namespace) -> object:
    """Build AdamW with ``anchor_lr`` and ``weight_decay``.

    Args:
        model (object): Model whose parameters are optimized.
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        torch.optim.AdamW: The optimizer.
    """
    return build_adamw(model, args, lr=args.anchor_lr, weight_decay=args.weight_decay)


def build_scheduler(
    optimizer: object, args: argparse.Namespace, last_epoch: int
) -> CosineWithWarmupScheduler:
    """Build the cosine-with-warmup LR scheduler (stepped every batch).

    Args:
        optimizer (object): Optimizer the scheduler wraps.
        args (argparse.Namespace): Parsed command-line arguments.
        last_epoch (int): Scheduler ``last_epoch`` for continuation.

    Returns:
        CosineWithWarmupScheduler: The learning-rate scheduler.
    """
    return CosineWithWarmupScheduler(
        optimizer,
        anchor_lr=args.anchor_lr,
        terminal_steps=args.terminal_steps,
        warmup_steps=args.warmup_steps,
        num_cycles=args.num_cycles,
        min_fraction=args.min_fraction,
        last_epoch=last_epoch,
    )


if __name__ == "__main__":
    args = parser.parse_args()

    if not 0 <= args.fold < args.n_folds:
        parser.error(f"--fold must be in [0, {args.n_folds}), got {args.fold}.")

    HarnessTrainer(
        args,
        model_builder=build_from_checkpoint(
            SwiGLU_mlp, make_model_args, optimizer_kwargs=optimizer_kwargs(args)
        ),
        dataset_builder=build_dataset,
        epoch_fn=train_DDP_array_epoch,
        optimizer_builder=build_optimizer,
        scheduler_builder=build_scheduler,
        loss_builder=functools.partial(VectorLpLoss, p=2, reduction="none"),
        device_type="cpu",
    ).run()
