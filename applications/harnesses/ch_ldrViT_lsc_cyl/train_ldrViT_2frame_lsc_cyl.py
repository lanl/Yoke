"""DDP harness training a 2-frame LodeRunner-ViT on LSC and CYL together.

Thin wrapper around :class:`yoke.harnesses.trainer.HarnessTrainer`, structured
like ``ch_ldrViT/train_ldrViT_2frame_8ch_lp.py``: same backbone, optimizer,
scheduler, grad clip, and EMA, and the same **per-channel** ``LpLoss``
(:class:`yoke.losses.lp_loss.LpLoss`, ``d=2, p=2, method="abs", eps=1e-4``)
in place of MSE. Unlike that script, density, energy and pressure are trained
for every material (``--thermo_kinds``); the per-channel norm balances their
very different scales.

The difference is the data: samples come from
:mod:`yoke.datasets.pagosa_2frame`, which maps both the LSC
(``lsc240420``, part-named fields) and Cylex (``cx241203``, partly
material-named fields) datasets onto the shared material vocabulary
:data:`~yoke.datasets.pagosa_fields.CANONICAL_FIELDS`. The model's
``default_vars`` is that vocabulary, and each sample carries its own channel
map, so batches must use ``--batch_size 1``.

``--lsc_weight`` and ``--cyl_weight`` set the mixing proportions; a weight of
``0`` trains on the other dataset alone. Loss records gain a fourth column
naming each sample's dataset (plot one with ``yoke-plot-loss --dataset lsc``).
Recorded losses are per-channel L2 norms averaged over channels, as in
``train_ldrViT_2frame_8ch_lp.py``.
"""

import argparse

from yoke.datasets.mixed_dataset import MixedDataSet
from yoke.datasets.pagosa_2frame import (
    CYL_2frame_canonical_DataSet,
    LSC_2frame_canonical_DataSet,
)
from yoke.datasets.pagosa_fields import CANONICAL_FIELDS, THERMO_KINDS
from yoke.harnesses.trainer import HarnessTrainer, TrainerHooks
from yoke.helpers import cli
from yoke.losses.lp_loss import LpLoss
from yoke.lr_schedulers import CosineWithWarmupScheduler
from yoke.models.vit.swin.bomberman import LodeRunnerViT
from yoke.utils.builders import build_adamw, build_from_checkpoint
from yoke.utils.ema import make_ema_hooks
from yoke.utils.training.epoch.loderunner import (
    train_DDP_loderunner_multidataset_epoch,
)

#############################################
# Inputs
#############################################
descr_str = (
    "Uses DDP to train a 2-frame LodeRunner-ViT on lsc240420 and cx241203 "
    "simultaneously using material-based canonical channels and a "
    "per-channel LpLoss."
)
parser = argparse.ArgumentParser(
    prog="DDP LodeRunner-ViT LSC+CYL Training (LpLoss)",
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
    help="Maximum file-index gap between consecutive frames.",
)

# Dataset mixing. LSC uses --LSC_NPZ_DIR; CYL uses --NPZ_DIR and --CSV_FILEPATH.
parser.add_argument(
    "--lsc_train_filelist",
    type=str,
    default="lsc240420_prefixes_train_80pct.txt",
    help="LSC training prefix list, relative to FILELIST_DIR.",
)
parser.add_argument(
    "--lsc_validation_filelist",
    type=str,
    default="lsc240420_prefixes_validation_10pct.txt",
    help="LSC validation prefix list, relative to FILELIST_DIR.",
)
parser.add_argument(
    "--cyl_train_filelist",
    type=str,
    default="cx241203_prefixes_train_80pct.txt",
    help="CYL training prefix list, relative to FILELIST_DIR.",
)
parser.add_argument(
    "--cyl_validation_filelist",
    type=str,
    default="cx241203_prefixes_val_10pct.txt",
    help="CYL validation prefix list, relative to FILELIST_DIR.",
)
parser.add_argument(
    "--lsc_weight",
    type=float,
    default=1.0,
    help="Relative sampling weight of LSC. Set 0 to train on CYL only.",
)
parser.add_argument(
    "--cyl_weight",
    type=float,
    default=1.0,
    help="Relative sampling weight of CYL. Set 0 to train on LSC only.",
)
parser.add_argument(
    "--thermo_kinds",
    type=str,
    nargs="+",
    choices=THERMO_KINDS,
    default=list(THERMO_KINDS),
    help="Per-material thermodynamic fields to train on.",
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


def make_model_args(args: argparse.Namespace) -> dict:
    """Build the 2-frame LodeRunner-ViT ``model_args`` over canonical channels.

    The backbone settings match ``ch_ldrViT/train_ldrViT_2frame.py``; only
    ``default_vars`` differs, being the full canonical PAGOSA vocabulary.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        dict: Keyword arguments for constructing :class:`LodeRunnerViT`.
    """
    return {
        "default_vars": list(CANONICAL_FIELDS),
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


def build_lp_loss() -> LpLoss:
    """Build the per-channel ``LpLoss`` used by the varchan datasteps.

    Identical to ``build_lp_loss`` in ``ch_ldrViT/train_ldrViT_2frame_8ch_lp.py``
    (UMich WAMRViT PLI ViT finest loss): an absolute per-channel L2 norm over
    the ``(H, W)`` axes with no averaging over field elements.
    ``reduction="none"`` returns the per-``(B, C)`` norm so the datasteps
    control the reduction (mean over batch and channel for backprop; mean over
    channel for the recorded per-sample loss).

    Returns:
        LpLoss: The per-channel loss module.
    """
    return LpLoss(d=2, p=2, method="abs", eps=1e-4, reduction="none")


def build_split(args: argparse.Namespace, split: str) -> MixedDataSet:
    """Build the mixed LSC/CYL dataset for one split.

    Datasets with weight ``0`` are not constructed, so their data paths need
    not exist.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        split (str): ``"train"`` or ``"validation"``.

    Returns:
        MixedDataSet: The weighted mixture.

    Raises:
        ValueError: If both weights are ``0``.
    """
    common = {
        "max_timeIDX_offset": args.max_timeIDX_offset,
        "max_file_checks": 10,
        "half_image": True,
        "thermo_kinds": args.thermo_kinds,
    }
    datasets, weights = [], []
    if args.lsc_weight > 0:
        filelist = getattr(args, f"lsc_{split}_filelist")
        datasets.append(
            LSC_2frame_canonical_DataSet(
                args.LSC_NPZ_DIR, args.FILELIST_DIR + filelist, **common
            )
        )
        weights.append(args.lsc_weight)
    if args.cyl_weight > 0:
        filelist = getattr(args, f"cyl_{split}_filelist")
        datasets.append(
            CYL_2frame_canonical_DataSet(
                args.NPZ_DIR,
                args.CSV_FILEPATH,
                args.FILELIST_DIR + filelist,
                **common,
            )
        )
        weights.append(args.cyl_weight)
    if not datasets:
        raise ValueError("At least one of --lsc_weight/--cyl_weight must be > 0.")
    return MixedDataSet(datasets, weights)


def build_dataset(args: argparse.Namespace) -> tuple[MixedDataSet, MixedDataSet]:
    """Build the train/validation mixed datasets.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        tuple: ``(train_dataset, val_dataset)``.
    """
    return build_split(args, "train"), build_split(args, "validation")


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


if __name__ == "__main__":
    args = parser.parse_args()

    # Samples carry their own channel maps, which differ between simulations,
    # so a batch can only hold samples sharing one map.
    if args.batch_size != 1:
        parser.error("--batch_size must be 1 for variable-channel training.")

    grad_clip = args.grad_clip if args.grad_clip and args.grad_clip > 0 else None
    epoch_kwargs = {
        "grad_clip": grad_clip,
        "ema_update_after_step": args.ema_update_after_step,
    }

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
        epoch_fn=train_DDP_loderunner_multidataset_epoch,
        optimizer_builder=build_optimizer,
        scheduler_builder=build_scheduler,
        loss_builder=build_lp_loss,
        epoch_kwargs=epoch_kwargs,
        hooks=hooks,
    ).run()
