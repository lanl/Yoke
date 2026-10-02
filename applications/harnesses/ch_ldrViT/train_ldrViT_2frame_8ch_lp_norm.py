"""DDP training harness for LodeRunner-ViT on lsc240420 (8ch, LpLoss, normalized).

Variant of ``train_ldrViT_2frame_8ch_lp.py`` that additionally applies the
per-channel input normalization scheme in
:mod:`yoke.datasets.lsc_normalization` via
:class:`~yoke.datasets.lsc_dataset.LSC_rho2rho_temporal_2frame_normalized_DataSet`,
in place of the un-normalized
:class:`~yoke.datasets.lsc_dataset.LSC_rho2rho_temporal_2frame_DataSet`.

Three changes relative to the baseline ``train_ldrViT_2frame.py``, all
discussed in
``misc_work_notes/ArtIMich/LodeRunnerViT_2Frame_improvement_plan.md`` and the
channel-normalization follow-up:

1. **8-channel target** -- the 6 material densities (``volfrac * density``)
   plus the two velocity components, instead of the full 20-channel
   density/energy/pressure set.
2. **Per-channel ``LpLoss``** (:class:`yoke.losses.lp_loss.LpLoss`, matching
   the UMich WAMRViT PLI ViT loss) in place of the plain ``nn.MSELoss`` used
   by the baseline, via the ``"pli_2frame_lp"`` datastep family.
3. **Per-channel input normalization**: a fixed, global, scale-only transform
   (density fields divided by a one-sided percentile-based scale so the
   material-absent background stays exactly ``0``; velocity fields divided by
   a symmetric percentile-based scale) computed once ahead of training by
   ``applications/normalization/generate_lsc_channel_normalization.py`` and
   applied by :class:`LSC_rho2rho_temporal_2frame_normalized_DataSet`. This is
   a network-input-conditioning fix (not a loss-balancing fix -- ``LpLoss``'s
   per-channel norm already handles that dynamically); see the normalization
   module docstring for the full rationale.

A new required CLI argument, ``--channel_norm_file``, points at the NPZ
produced by ``generate_lsc_channel_normalization.py`` for the 8-channel set.

Everything else -- backbone settings matched to ArtIMich ViT (finest), the
``anchor_lr`` AdamW optimizer, the cosine-with-warmup LR scheduler, gradient
clipping, and the Diffusers-style warmup EMA shadow -- is identical to
``train_ldrViT_2frame_8ch_lp.py``.
"""

import argparse

import numpy as np

from yoke.datasets.lsc_dataset import LSC_rho2rho_temporal_2frame_normalized_DataSet
from yoke.harnesses.trainer import HarnessTrainer, TrainerHooks
from yoke.helpers import cli
from yoke.losses.lp_loss import LpLoss
from yoke.lr_schedulers import CosineWithWarmupScheduler
from yoke.models.vit.swin.bomberman import LodeRunnerViT
from yoke.utils.builders import build_adamw, build_from_checkpoint
from yoke.utils.ema import make_ema_hooks
from yoke.utils.training.epoch.loderunner import train_DDP_loderunner_epoch

#############################################
# Inputs
#############################################
descr_str = (
    "Uses DDP to train LodeRunner-ViT architecture on lsc240420 with an "
    "8-channel target, a per-channel LpLoss, and per-channel input "
    "normalization."
)
parser = argparse.ArgumentParser(
    prog="DDP LodeRunner-ViT Training (8ch, LpLoss, normalized)",
    description=descr_str,
    fromfile_prefix_chars="@",
)
parser = cli.add_default_args(parser=parser)
parser = cli.add_filepath_args(parser=parser)
parser = cli.add_computing_args(parser=parser)
parser = cli.add_model_args(parser=parser)
parser = cli.add_training_args(parser=parser)
parser = cli.add_cosine_lr_scheduler_args(parser=parser)

parser.add_argument(
    "--max_timeIDX_offset",
    type=int,
    default=1,
    help="Maximum time offset for input/output image pairs.",
)

# ViT backbone parameters
parser.add_argument(
    "--vit_embed_dim",
    type=int,
    default=512,
    help="Embedding dimension for the ViT backbone.",
)
parser.add_argument(
    "--vit_num_layers",
    type=int,
    default=12,
    help="Number of ViT layers in backbone.",
)
parser.add_argument(
    "--vit_num_heads",
    type=int,
    default=8,
    help="Number of ViT attention heads in backbone.",
)

# EMA + gradient-clipping parameters (ArtIMich recipe).
parser.add_argument(
    "--use_ema",
    action="store_true",
    default=True,
    help="Maintain a Diffusers-style warmup EMA shadow of the model and save "
    "the EMA weights as a companion production checkpoint.",
)
parser.add_argument(
    "--ema_update_after_step",
    type=int,
    default=1000,
    help="Global optimizer-step after which the EMA begins updating.",
)
parser.add_argument(
    "--ema_max_decay",
    type=float,
    default=0.9999,
    help="Maximum (asymptotic) EMA decay.",
)
parser.add_argument(
    "--ema_inv_gamma",
    type=float,
    default=1.0,
    help="Inverse-gamma factor controlling the EMA warmup rate.",
)
parser.add_argument(
    "--ema_power",
    type=float,
    default=2.0 / 3.0,
    help="Warmup power for the EMA decay schedule.",
)
parser.add_argument(
    "--grad_clip",
    type=float,
    default=1.0,
    help="Global gradient-norm clip value applied before each optimizer step. "
    "Set <= 0 to disable clipping.",
)

# Per-channel input normalization.
parser.add_argument(
    "--channel_norm_file",
    type=str,
    required=True,
    help="Path to the per-channel normalization NPZ produced by "
    "applications/normalization/generate_lsc_channel_normalization.py. Must "
    "contain a normalization record for every channel in CHANNEL_LIST.",
)

# Change some default filepaths.
parser.set_defaults(
    train_filelist="lsc240420_prefixes_train_80pct.txt",
    validation_filelist="lsc240420_prefixes_validation_10pct.txt",
    test_filelist="lsc240420_prefixes_test_10pct.txt",
)

#############################################
# Study-specific constants
#############################################
# 8-channel target: the 6 material densities (volfrac * density) plus the two
# velocity components. This matches
# LSC_rho2rho_temporal_2frame_DataSet's default `hydro_fields`.
CHANNEL_LIST = [
    "density_case",
    "density_cushion",
    "density_maincharge",
    "density_outside_air",
    "density_striker",
    "density_throw",
    "Uvelocity",
    "Wvelocity",
]


def make_model_args(args: argparse.Namespace) -> dict:
    """Build the 2-frame LodeRunner-ViT ``model_args`` dict from parsed args.

    The ArtIMich ViT (finest) backbone (ignoring its data pipeline / channel
    layout, which is **not** reproduced here) is defined by ``num_layers=6``,
    ``num_attention_heads=12``, ``attention_head_dim=192`` (embedding dim =
    heads x head_dim), ``mlp_ratio=1.0``, ``rope_theta=10000``,
    ``rope_scale=[80, 224]``, ``eps=1e-7``, and a ``[10, 5]`` spatial patch on
    the 1120x400 image (112 x 80 = 8960 spatial tokens).

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        dict: Keyword arguments for constructing :class:`LodeRunnerViT`.
    """
    return {
        "default_vars": CHANNEL_LIST,
        "image_size": (1120, 400),
        "patch_size": (10, 5),
        "embed_dim": args.vit_embed_dim,
        "num_heads": 8,
        "num_attention_heads": args.vit_num_heads,
        "attention_head_dim": int(args.vit_embed_dim / args.vit_num_heads),
        "num_layers": args.vit_num_layers,
        "rope_theta": 10000,
        "rope_scale": (80, 224),
        "mlp_ratio": 1.0,
        "concat_mlp": True,
        "verbose": False,
        "num_input_frames": 2,
        "eps": 1e-7,
        "bias": True,
    }


def build_optimizer(model: object, args: argparse.Namespace) -> object:
    """Build the AdamW optimizer using ``anchor_lr`` as the fresh-run LR.

    Args:
        model (object): Model whose parameters are optimized.
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        torch.optim.AdamW: The optimizer.
    """
    return build_adamw(model, args, lr=args.anchor_lr, weight_decay=0.01)


def build_dataset(args: argparse.Namespace) -> tuple[object, object]:
    """Build the train/validation normalized 2-frame LSC temporal datasets.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        tuple: ``(train_dataset, val_dataset)``.
    """
    train_filelist = args.FILELIST_DIR + args.train_filelist
    validation_filelist = args.FILELIST_DIR + args.validation_filelist

    train_dataset = LSC_rho2rho_temporal_2frame_normalized_DataSet(
        args.LSC_NPZ_DIR,
        file_prefix_list=train_filelist,
        max_timeIDX_offset=args.max_timeIDX_offset,
        max_file_checks=10,
        channel_norm_file=args.channel_norm_file,
        hydro_fields=np.array(CHANNEL_LIST),
        half_image=True,
    )
    val_dataset = LSC_rho2rho_temporal_2frame_normalized_DataSet(
        args.LSC_NPZ_DIR,
        file_prefix_list=validation_filelist,
        max_timeIDX_offset=args.max_timeIDX_offset,
        max_file_checks=10,
        channel_norm_file=args.channel_norm_file,
        hydro_fields=np.array(CHANNEL_LIST),
        half_image=True,
    )
    return train_dataset, val_dataset


def build_scheduler(
    optimizer: object, args: argparse.Namespace, last_epoch: int
) -> CosineWithWarmupScheduler:
    """Build the cosine-with-warmup LR scheduler.

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


def build_lp_loss() -> LpLoss:
    """Build the per-channel ``LpLoss`` used by the ``pli_2frame_lp`` datasteps.

    Matches the UMich WAMRViT PLI ViT (finest) loss: an absolute per-channel
    L2 norm (``p=2``, ``method="abs"``, ``eps=1e-4``) with no averaging over
    the flattened field elements. ``d=2`` norms over the trailing ``(H, W)``
    axes of the ``(B, C, H, W)`` 2-frame prediction. ``reduction="none"``
    returns the per-``(B, C)`` norm so the ``..._2frame_lp_datastep`` functions
    control the final backprop/recording reduction (mean over batch and
    channel for backprop; mean over channel for the recorded per-sample loss).

    Returns:
        LpLoss: The per-channel loss module.
    """
    return LpLoss(d=2, p=2, method="abs", eps=1e-4, reduction="none")


if __name__ == "__main__":
    args = parser.parse_args()

    # Gradient clipping and EMA-update cadence are handled inside the epoch
    # function; the grad-clip threshold is threaded through epoch_kwargs. The
    # "pli_2frame_lp" tag selects the LpLoss-aware two-frame datastep functions.
    grad_clip = args.grad_clip if args.grad_clip and args.grad_clip > 0 else None
    epoch_kwargs = {
        "channel_map": list(range(len(CHANNEL_LIST))),
        "dataset": "pli_2frame_lp",
        "grad_clip": grad_clip,
        "ema_update_after_step": args.ema_update_after_step,
    }

    # Optionally maintain a warmup-EMA shadow via the shared hooks.
    hooks = None
    if args.use_ema:
        on_after_ddp_wrap, on_before_save = make_ema_hooks(
            LodeRunnerViT,
            max_decay=args.ema_max_decay,
            inv_gamma=args.ema_inv_gamma,
            power=args.ema_power,
        )
        hooks = TrainerHooks(
            on_after_ddp_wrap=on_after_ddp_wrap,
            on_before_save=on_before_save,
        )

    HarnessTrainer(
        args,
        model_builder=build_from_checkpoint(
            LodeRunnerViT,
            make_model_args,
            optimizer_kwargs={
                "lr": args.anchor_lr,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0.01,
            },
        ),
        dataset_builder=build_dataset,
        epoch_fn=train_DDP_loderunner_epoch,
        optimizer_builder=build_optimizer,
        scheduler_builder=build_scheduler,
        loss_builder=build_lp_loss,
        epoch_kwargs=epoch_kwargs,
        hooks=hooks,
    ).run()
