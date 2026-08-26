import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import ConcatDataset

from yoke.models.vit.swin.bomberman import (
    LodeRunner,
    ScalarTemporalConditionedLodeRunner_9band,
)
from yoke.datasets.kilonova_dataset import (
    Kilonova_lc_scalar_context_DataSet_9band,
    NINE_BAND_KEYS,
    load_or_compute_band_normalization,
)
from yoke.utils.training.epoch.loderunner import (
    train_DDP_scalar_temporal_loderunner_epoch_9band,
    train_DDP_scalar_temporal_loderunner_epoch_9band_rollout,
)
from yoke.utils.ema import ParamEMA
from yoke.utils.restart import continuation_setup
from yoke.utils.dataload import make_distributed_dataloader
from yoke.utils.checkpointing import load_model_and_optimizer
from yoke.utils.checkpointing import save_model_and_optimizer
from yoke.utils.checkpointing import load_direct_loderunner_checkpoint_9band
from yoke.utils.parallel import setup_distributed, cleanup_distributed
from yoke.lr_schedulers import CosineWithWarmupScheduler
from yoke.helpers import cli

def _read_stem_list(path: str) -> set:
    """Read an object-stem split file (one stem per line) into a set.

    Deterministic and RNG-free, so every DDP rank builds the identical set.
    """
    with open(path) as fh:
        return {line.strip() for line in fh if line.strip()}


#############################################
# Inputs
#############################################
descr_str = (
    "Uses DDP to train LodeRunner architecture on single-timstep input and output "
    "of the lsc240420 per-material density fields."
)
parser = argparse.ArgumentParser(
    prog="DDP LodeRunner Training", description=descr_str, fromfile_prefix_chars="@"
)
parser = cli.add_default_args(parser=parser)
parser = cli.add_filepath_args(parser=parser)
parser = cli.add_computing_args(parser=parser)
parser = cli.add_model_args(parser=parser)
parser = cli.add_training_args(parser=parser)
parser = cli.add_cosine_lr_scheduler_args(parser=parser)

# DPOT‐style noise parameter
parser.add_argument(
    "--noise_scale",
    type=float,
    default=0.0,
    help="Relative magnitude ε for Gaussian noise injection (e.g. 5e-5).",
)

# Multi-step rollout training (scheduled sampling) to address exposure bias.
parser.add_argument(
    "--n_rollout_steps",
    type=int,
    default=1,
    help="Number of future events supervised per sample. 1 uses the standard "
    "single-step teacher-forced training; >1 enables scheduled-sampling "
    "rollout training.",
)
parser.add_argument(
    "--tf_start",
    type=float,
    default=1.0,
    help="Teacher-forcing ratio at epoch 0 (probability of feeding the true "
    "value back at each rollout step). Only used when --n_rollout_steps > 1.",
)
parser.add_argument(
    "--tf_end",
    type=float,
    default=0.0,
    help="Teacher-forcing ratio the schedule anneals down to. Only used when "
    "--n_rollout_steps > 1.",
)
parser.add_argument(
    "--tf_ramp_epochs",
    type=int,
    default=50,
    help="Number of epochs over which the teacher-forcing ratio decays linearly "
    "from --tf_start to --tf_end. Only used when --n_rollout_steps > 1.",
)
parser.add_argument(
    "--tf_ramp_start_epoch",
    type=int,
    default=0,
    help="Absolute epoch at which the teacher-forcing anneal begins. Because "
    "epoch numbering continues across restarts, set this to the checkpoint "
    "epoch when starting rollout training as a continuation (e.g. 50 when "
    "continuing from epoch 50) so the ramp is measured from there rather than "
    "from epoch 0. Only used when --n_rollout_steps > 1.",
)

# Per-step weight EMA (Polyak averaging) of the trainable params (conditioner +
# output_head). A shadow average that survives the cycle_epochs=1 process restart
# through the checkpoint. Smooths the jagged per-batch trajectory; eval can opt in
# via --use_ema.
parser.add_argument(
    "--ema_decay",
    type=float,
    default=0.999,
    help="EMA decay for the trainable-parameter shadow (Polyak averaging), "
    "updated after each optimizer step. Higher = slower/smoother. Set 0 to "
    "disable EMA entirely.",
)

# Paired-dataset globs. The realistic set is always used; the dense set (same
# objects, denser cadence, no limiting-mag cut) is optional and concatenated onto
# the realistic training data when present. Both are filtered to the object-level
# split (see --train_filelist / --validation_filelist below).
parser.add_argument(
    "--kn_realistic_glob",
    type=str,
    default=(
        "/net/sescratch1/atoivonen/data/KN_lightcurves/"
        "rubin_ztf_10000_dataset_same_seed/lc_*.npz"
    ),
    help="Glob for the realistic light-curve files (primary training data).",
)
parser.add_argument(
    "--kn_dense_glob",
    type=str,
    default=(
        "/net/sescratch1/atoivonen/data/KN_lightcurves/"
        "rubin_ztf_dense_10000_dataset_same_seed/lc_*.npz"
    ),
    help="Optional glob for the dense light-curve files. When set (and it "
    "matches files), the dense TRAIN objects are concatenated onto the "
    "realistic TRAIN objects to supervise late-time behavior. Validation and "
    "the primary metric stay realistic-only.",
)

# Change some default filepaths. The KN split lists hold object stems (one per
# line), shared across the realistic and dense directories; see
# make_kn_object_lists.py.
parser.set_defaults(
    train_filelist="kn_rubin_ztf_train.txt",
    validation_filelist="kn_rubin_ztf_val.txt",
    test_filelist="kn_rubin_ztf_test.txt",
)


def main(args, rank, world_size, local_rank, device):
    #############################################
    # Process Inputs
    #############################################
    # Study ID
    studyIDX = args.studyIDX

    # Resources
    Ngpus = args.Ngpus
    Knodes = args.Knodes

    # Data Paths. The KN train/val/test lists hold object stems (one per line);
    # the same stems select objects in both the realistic and dense directories,
    # so an object is entirely in train or entirely in val/test in both views.
    train_filelist = args.FILELIST_DIR + args.train_filelist
    validation_filelist = args.FILELIST_DIR + args.validation_filelist
    train_stems = _read_stem_list(train_filelist)
    val_stems = _read_stem_list(validation_filelist)

    # Model Parameters
    embed_dim = args.embed_dim
    block_structure = tuple(args.block_structure)

    # Training Parameters
    anchor_lr = args.anchor_lr
    num_cycles = args.num_cycles
    min_fraction = args.min_fraction
    terminal_steps = args.terminal_steps
    warmup_steps = args.warmup_steps
    noise_scale = args.noise_scale

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

    #############################################
    # Model Arguments for Dynamic Reconstruction
    #############################################
    # Dictionary of available models.
    available_models = {
        "LodeRunner": LodeRunner
    }

    # Model arguments for LodeRunner.
    model_args = {
        "default_vars": [
            "density_case",
            "density_cushion",
            "density_maincharge",
            "density_outside_air",
            "density_striker",
            "density_throw",
            "Uvelocity",
            "Wvelocity",
        ],
        "image_size": (1120, 400),
        "patch_size": (10, 5),
        "embed_dim": embed_dim,
        "emb_factor": 2,
        "num_heads": 8,
        "block_structure": block_structure,
        "window_sizes": [(8, 8), (8, 8), (4, 4), (2, 2)],
        "patch_merge_scales": [(2, 2), (2, 2), (2, 2)],
        #"noise_scale": noise_scale,
    }


    CONTEXT_LEN = 5 #3
    HIDDEN_CHANNELS = 64

    # Fourier lead-time conditioning. When > 0, the trainable conditioner and
    # output head receive a 2*DT_FOURIER_BANDS sinusoidal encoding of the lead
    # time Dt, so they can learn a real per-band decay curve instead of a flat
    # persistence value. (Without this, Dt reaches the output only through the
    # frozen backbone, which cannot adapt, so late-time forecasts plateau.) Set
    # to 0 for the legacy architecture (byte-identical; old checkpoints load).
    DT_FOURIER_BANDS = 8

    # Delta-anchored head. When True, the output head predicts a CHANGE relative
    # to the per-band last observed magnitude (fallback: most-recent observation
    # in any band) instead of an absolute magnitude, so the forecast starts AT the
    # last observation at Dt=0 rather than reconstructing the zero-point from
    # scratch. This kills the ~0.5-0.8 mag lead-0 offset seen in the dense eval.
    # Adds NO parameters (the anchor is derived from x), so old checkpoints still
    # load strict=True; the flag is saved and restored by the loaders. Requires
    # window mode (needs the per-event validity flag). Set False for the absolute
    # head (byte-identical numerics to the pre-delta model).
    PREDICT_DELTA = True

    # Trend/decay anchor (delta head only). When True, the per-band anchor the
    # head predicts a residual on top of is no longer the flat last-observed value
    # but a locally-extrapolated one: anchor[b] = v_last[b] + slope[b] * Dt, with
    # slope from a least-squares fit over the band's most-recent TREND_SLOPE_K
    # valid events. This lets the forecast LEAN INTO the fade instead of holding
    # flat -- targeting the residual blue-band under-fade (u/ztfg/g) that the
    # flat-hold anchor structurally cannot track. The slope is RAW (unclamped):
    # bands sampled pre-peak can extrapolate continued brightening (accepted
    # risk; the head learns a residual on top). To forbid brightening, clamp the
    # slope non-negative in _band_anchor (one line, flagged there). Adds NO
    # parameters (derived from x/Dt), so old checkpoints load strict=True; the
    # flags round-trip via the loaders. Requires PREDICT_DELTA + window mode.
    TREND_DECAY_ANCHOR = True
    TREND_SLOPE_K = 3

    # Per-step weight EMA (Polyak averaging) decay for the trainable params. Read
    # from --ema_decay so it flows through the @input file and survives resubmits.
    # 0 disables EMA. The shadow is saved into and restored from the .pth each
    # epoch so it survives the cycle_epochs=1 process restart.
    EMA_DECAY = args.ema_decay

    # Time-window context mode. When CONTEXT_WINDOW_DAYS is not None, the dataset
    # selects context by a trailing lookback in days (all detections within the
    # last CONTEXT_WINDOW_DAYS), padded to MAX_CONTEXT_LEN with a per-event
    # validity flag, instead of a fixed count of CONTEXT_LEN events. This gives
    # the model real time evolution instead of a single dense night. Set these
    # from the plot_observation_histograms.py time-window sweep. Leave
    # CONTEXT_WINDOW_DAYS = None to use the legacy fixed-count context.
    CONTEXT_WINDOW_DAYS = 2.0
    MAX_CONTEXT_LEN = 12

    # Horizon-covering target sampling (window mode only). When set, each sample
    # draws its target lead time ~uniform in days over (0, TARGET_HORIZON_DAYS]
    # and supervises the event nearest that lead time, instead of always the
    # immediate next event. This flattens the supervised-Dt distribution so the
    # model is trained at the multi-day lead times it is asked to forecast,
    # rather than only at the short gap-to-next-event (p99 ~4d on this set) while
    # rollouts forecast out to ~12d. Set from the plot_observation_histograms.py
    # "Supervised Δt vs forecast horizon" panel (the h=1 tail is the uncovered
    # region). Leave None to supervise the immediate next event as before.
    TARGET_HORIZON_DAYS = 8.0
    # In window mode the model's first layer is sized by the padded width.
    WRAPPER_CONTEXT_LEN = (
        MAX_CONTEXT_LEN if CONTEXT_WINDOW_DAYS is not None else CONTEXT_LEN
    )

    # Multi-step rollout training config (scheduled sampling). n_rollout_steps=1
    # falls back to the standard single-step teacher-forced training.
    n_rollout_steps = args.n_rollout_steps
    tf_start = args.tf_start
    tf_end = args.tf_end
    tf_ramp_epochs = max(1, args.tf_ramp_epochs)
    tf_ramp_start_epoch = args.tf_ramp_start_epoch

    # Nine-band merged event-stream setup (3 ZTF + 6 Rubin/LSST bands).
    BAND_KEYS = NINE_BAND_KEYS
    VALUE_COL = 1
    ERROR_COL = 2
    N_BANDS = len(BAND_KEYS)

    # Upper-limit (non-detection) observations are flagged by a non-finite
    # uncertainty in ERROR_COL. Drop them so the model trains only on real
    # detections; normalization statistics are computed the same way.
    DROP_UPPER_LIMITS = True

    # Per-band loss weighting. Targets are per-band z-scored, so an equal-weight
    # loss lets the large-dynamic-range blue bands (u, g fade to mag ~28-30) be
    # under-fit -- the dense-eval showed a strong under-fade bias there (u
    # ~ -5 mag, g ~ -2 mag). Up-weighting these bands in the TRAINING backward
    # (the recorded per-sample loss stays unweighted so the val CSV remains a
    # comparable yardstick) pushes gradient toward the fades the model currently
    # ignores. The three ZTF bands are also up-weighted (1->2): in autoregressive
    # rollout they lag because ZTF is realistically sampled near peak/early with
    # little late-time context to re-anchor, so they get the least gradient
    # pressure exactly where they fail. Order matches BAND_KEYS =
    # (ztfg, ztfr, ztfi, sdssu, ps1_g, ps1_r, ps1_i, ps1_z, ps1_y). Set to None
    # to recover the exact equal-weight objective.
    BAND_WEIGHTS = torch.tensor(
        [
            2.0,  # ztfg -- ZTF bands lag in rollout; up-weight from 1->2
            2.0,  # ztfr -- ZTF bands lag in rollout; up-weight from 1->2
            2.0,  # ztfi -- ZTF bands lag in rollout; up-weight from 1->2
            3.0,  # sdssu (u) -- worst under-fade, largest up-weight
            2.0,  # ps1__g (g)
            1.0,  # ps1__r (r)
            1.0,  # ps1__i (i)
            1.0,  # ps1__z (z)
            1.0,  # ps1__y (y)
        ],
        dtype=torch.float32,
    )

    # Lead-time loss weighting (window-mode rollout only). The dense truth has far
    # more points in the slowly-fading late tail than in the 2-5 d rise, so an
    # equal-per-step objective rewards nailing the flat tail and the forecast
    # collapses to a plateau. Weighting each rollout step by 1 / (1 + Dt / tau)
    # gives short-lead steps (the rise, where the model fails) relatively more
    # gradient. Composes multiplicatively with BAND_WEIGHTS. The recorded
    # per-sample loss stays unweighted so the val CSV remains comparable. Set to
    # None to recover the equal-per-step objective. tau ~ a few days.
    DT_WEIGHT_TAU = 3.0

    optimizer_kwargs = {
        "lr": 1e-4,# 1e-4, #1e-5
        "betas": (0.9, 0.999),
        "eps": 1e-08,
        "weight_decay": 0.01,
    }


    if CONTINUATION:
        model, optimizer, starting_epoch = load_direct_loderunner_checkpoint_9band(
            checkpoint_path=checkpoint,
            model_args=model_args,
            optimizer_kwargs=optimizer_kwargs,
            device=device,
        )

        if rank == 0:
            print(f"Loaded direct checkpoint from {checkpoint}")
            print(f"Continuing from epoch {starting_epoch}")

        '''  # FIXME block should be unindented if uncommented 
        if CONTINUATION:
            model, optimizer, starting_epoch = load_model_and_optimizer(
                checkpoint,
                optimizer_class=torch.optim.AdamW,
                optimizer_kwargs=optimizer_kwargs,
                available_models=available_models,
                device=device,
            )

            if rank == 0:
                print(f"Loaded temporal checkpoint from {checkpoint}")
                print(f"Continuing from epoch {starting_epoch}")
        '''

    else:
        starting_epoch = 0

        model = LodeRunner(**model_args)
        model.to(device)

        manual_checkpoint = "/usr/projects/artimis/mpmm/pretrained_models/ddp_ldr_prod_250721/study005_modelState_epoch0100.pth"

        checkpoint_data = torch.load(
            manual_checkpoint,
            map_location=device,
            weights_only=False,
        )

        state_dict = checkpoint_data["model_state_dict"]

        if all(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

        missing_keys, unexpected_keys = model.load_state_dict(
            state_dict,
            strict=False,
        )

        if rank == 0:
            print("Loaded pretrained backbone weights.")
            print("Missing keys:", missing_keys)
            print("Unexpected keys:", unexpected_keys)

        # NOTE: model reconstruction and optimizer creation must run on ALL
        # ranks. If gated behind `if rank == 0:` the non-zero ranks keep the bare
        # LodeRunner and never define `optimizer`, which crashes DDP wrapping /
        # the LR scheduler on multi-GPU runs.
        model.noise_scale = noise_scale

        backbone = model

        model = ScalarTemporalConditionedLodeRunner_9band(
            backbone=backbone,
            context_len=WRAPPER_CONTEXT_LEN,
            n_bands=N_BANDS,
            image_size=model_args["image_size"],
            backbone_channels=8,
            hidden=HIDDEN_CHANNELS,
            context_window_days=CONTEXT_WINDOW_DAYS,
            dt_fourier_bands=DT_FOURIER_BANDS,
            predict_delta=PREDICT_DELTA,
            trend_decay_anchor=TREND_DECAY_ANCHOR,
            trend_slope_k=TREND_SLOPE_K,
        ).to(device)

        # Stage 1: freeze pretrained LodeRunner, train only conditioner + output head
        for p in model.backbone.parameters():
            p.requires_grad = False

        for p in model.conditioner.parameters():
            p.requires_grad = True

        for p in model.output_head.parameters():
            p.requires_grad = True

        optimizer = torch.optim.AdamW(
            list(model.conditioner.parameters()) +
            list(model.output_head.parameters()),
            **optimizer_kwargs,
        )

    #loss_fn = nn.MSELoss(reduction="none")
    loss_fn = nn.HuberLoss(delta=0.1, reduction="none")
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    #############################################
    # Per-step weight EMA (Polyak averaging)
    #############################################
    # Shadow the trainable params only (conditioner + output_head; the frozen
    # backbone is excluded by the requires_grad filter). Keyed by name so it
    # round-trips through the .pth and survives the cycle_epochs=1 restart. On a
    # CONTINUATION, restore the shadow from the checkpoint BEFORE training so the
    # average is not silently reset to the current weights every epoch.
    ema = None
    if EMA_DECAY > 0:
        ema = ParamEMA(
            (
                (n, p)
                for n, p in model.module.named_parameters()
                if p.requires_grad
            ),
            decay=EMA_DECAY,
        )
        if CONTINUATION:
            ema_ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
            ema_sd = ema_ckpt.get("ema_state_dict")
            if ema_sd is not None:
                ema.load_state_dict(ema_sd)
                if rank == 0:
                    print(
                        f"Restored EMA shadow ({len(ema_sd)} tensors) from checkpoint."
                    )
            elif rank == 0:
                print(
                    "No ema_state_dict in checkpoint; EMA initialized from current "
                    "weights (expected on the first EMA-enabled epoch)."
                )

    #############################################
    # Learning Rate Scheduler
    #############################################
    print("Starting epoch: ", starting_epoch)
    if starting_epoch == 0:
        last_epoch = -1
    else:
        last_epoch = train_batches * (starting_epoch - 1)

    # Scale the anchor LR by global batchsize
    #
    # # For multi-node
    lr_scale = np.sqrt(float(Ngpus) * float(Knodes) * float(batch_size))
    original_batchsize = 40.0  # 1 node, 4 gpus, 10 samples/gpu
    ddp_anchor_lr = anchor_lr * lr_scale / original_batchsize
    #
    # For single node
    # ddp_anchor_lr = anchor_lr

    LRsched = LambdaLR(
        optimizer,
        lr_lambda=lambda step: 1.0,
        last_epoch=last_epoch,
    )


    LRsched = CosineWithWarmupScheduler(
        optimizer,
        anchor_lr=ddp_anchor_lr,
        terminal_steps=terminal_steps,
        warmup_steps=warmup_steps,
        num_cycles=num_cycles,
        min_fraction=min_fraction,
        last_epoch=last_epoch,
    )

    #############################################
    # Data Initialization (Distributed Dataloader)
    #############################################
    #train_dataset = LSC_rho2rho_temporal_DataSet(
    #    args.LSC_NPZ_DIR,
    #    file_prefix_list=train_filelist,
    #    max_timeIDX_offset=2,
    #    max_file_checks=10,
    #    half_image=True,
    #)
    #val_dataset = LSC_rho2rho_temporal_DataSet(
    #    args.LSC_NPZ_DIR,
    #    file_prefix_list=validation_filelist,
    #    max_timeIDX_offset=2,
    #    max_file_checks=10,
    #    half_image=True,
    #)

    '''

    train_dataset = Kilonova_lc_img_DataSet_channels_context(
        half_image=False,
        context_len=CONTEXT_LEN,
        #N_imgs=100,
    )

    val_dataset = Kilonova_lc_img_DataSet_channels_context(
        half_image=False,
        context_len=CONTEXT_LEN,
        #N_imgs=20, #100,
    )
    '''
    '''
    train_dataset = Kilonova_lc_scalar_context_DataSet(
        context_len=CONTEXT_LEN,
    )

    val_dataset = Kilonova_lc_scalar_context_DataSet(
        context_len=CONTEXT_LEN,
    )
    '''

    # Normalization statistics are computed over the TRAIN objects only (of the
    # realistic set) to avoid val/test leakage. The stats path is distinct from
    # the old all-files cache so a stale/leaky cache can't be silently reused.
    norm_stats_path = "kilonova_9band_norm_stats_trainonly.npz"
    train_norm_files = sorted(
        f
        for f in glob.glob(args.kn_realistic_glob)
        if os.path.splitext(os.path.basename(f))[0] in train_stems
    )

    if rank == 0:
        band_means, band_stds = load_or_compute_band_normalization(
            stats_path=norm_stats_path,
            band_keys=BAND_KEYS,
            value_col=VALUE_COL,
            error_col=ERROR_COL,
            drop_upper_limits=DROP_UPPER_LIMITS,
            file_prefix_list=train_norm_files,
        )

    dist.barrier()

    if rank != 0:
        stats = np.load(norm_stats_path, allow_pickle=True)
        band_means = stats["means"].astype(np.float32)
        band_stds = stats["stds"].astype(np.float32)
        stats.close()

    if rank == 0:
        print("Using band normalization:")
        print("band_means:", band_means)
        print("band_stds:", band_stds)

    # Object-level split makes every DDP rank build an identical-length dataset:
    # the train/val stem sets come from static files (no RNG), the file list is
    # sorted(glob(...)) then filtered by stem, and N_imgs=0 uses all matched
    # files (no np.random.choice). So no per-rank subset desync is possible.
    # (The remaining random.shuffle inside the dataset only reorders files; it
    # does not change the sample count.) Seed once for reproducibility only --
    # this is no longer load-bearing for rank sync. Keep N_imgs=0; N_imgs>0 would
    # reintroduce the unseeded-choice desync.
    DATA_SEED = 42
    np.random.seed(DATA_SEED)
    random.seed(DATA_SEED)

    def _make_9band(
        data_glob: str, object_ids: set
    ) -> Kilonova_lc_scalar_context_DataSet_9band:
        """Build a 9-band dataset over one directory restricted to object_ids."""
        return Kilonova_lc_scalar_context_DataSet_9band(
            N_imgs=0,
            context_len=CONTEXT_LEN,
            band_keys=BAND_KEYS,
            value_col=VALUE_COL,
            error_col=ERROR_COL,
            drop_upper_limits=DROP_UPPER_LIMITS,
            means=band_means,
            stds=band_stds,
            n_rollout_steps=n_rollout_steps,
            context_window_days=CONTEXT_WINDOW_DAYS,
            max_context_len=MAX_CONTEXT_LEN,
            target_horizon_days=TARGET_HORIZON_DAYS,
            data_glob=data_glob,
            object_ids=object_ids,
        )

    # Realistic TRAIN objects (always present) plus, if a dense set is provided
    # and matches files, the SAME train objects viewed densely -- concatenated to
    # supervise late-time behavior. Validation stays realistic-only (matches the
    # deployment metric).
    train_real = _make_9band(args.kn_realistic_glob, train_stems)
    train_parts = [train_real]

    if args.kn_dense_glob and glob.glob(args.kn_dense_glob):
        train_dense = _make_9band(args.kn_dense_glob, train_stems)
        if len(train_dense) > 0:
            train_parts.append(train_dense)
            if rank == 0:
                print(
                    f"Dense training set added: {len(train_dense)} samples "
                    f"(realistic: {len(train_real)} samples).",
                    flush=True,
                )
        elif rank == 0:
            print(
                "Dense glob matched files but yielded 0 samples for the train "
                "split; training on realistic set only.",
                flush=True,
            )
    elif rank == 0 and args.kn_dense_glob:
        print(
            "Dense glob set but matched no files; training on realistic set "
            "only.",
            flush=True,
        )

    train_dataset = (
        ConcatDataset(train_parts) if len(train_parts) > 1 else train_parts[0]
    )

    val_dataset = _make_9band(args.kn_realistic_glob, val_stems)


    # NOTE: For DDP the batch_size is the per-GPU batch_size!!!
    train_dataloader = make_distributed_dataloader(
        train_dataset,
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        rank=rank,
        world_size=world_size,
    )
    val_dataloader = make_distributed_dataloader(
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
        print('%%%%%%%%%%%%%')
        print(epochIDX)
        print('%%%%%%%%%%%%%')
        train_sampler = train_dataloader.sampler
        train_sampler.set_epoch(epochIDX)

        # For timing epochs
        if TIME_EPOCH:
            # Synchronize before starting the timer
            #dist.barrier()  # Ensure that all nodes sync
            torch.cuda.synchronize(device)  # Ensure GPUs on each node sync
            # Time each epoch and print to stdout
            startTime = time.time()


        if n_rollout_steps > 1:
            # Linearly anneal the teacher-forcing ratio from tf_start to tf_end
            # over tf_ramp_epochs, then hold at tf_end. The ramp is measured from
            # tf_ramp_start_epoch so it works correctly on a continuation, where
            # epoch numbering carries over from the previous run.
            frac = max(
                0.0,
                min(
                    1.0,
                    (epochIDX - 1 - tf_ramp_start_epoch) / tf_ramp_epochs,
                ),
            )
            teacher_forcing_ratio = tf_start + (tf_end - tf_start) * frac

            if rank == 0:
                print(
                    f"Rollout training: n_rollout_steps={n_rollout_steps}, "
                    f"teacher_forcing_ratio={teacher_forcing_ratio:.4f}",
                    flush=True,
                )

            train_DDP_scalar_temporal_loderunner_epoch_9band_rollout(
                training_data=train_dataloader,
                validation_data=val_dataloader,
                num_train_batches=train_batches,
                num_val_batches=val_batches,
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                LRsched=LRsched,
                epochIDX=epochIDX,
                train_per_val=train_per_val,
                train_rcrd_filename=trn_rcrd_filename,
                val_rcrd_filename=val_rcrd_filename,
                device=device,
                rank=rank,
                world_size=world_size,
                n_bands=N_BANDS,
                teacher_forcing_ratio=teacher_forcing_ratio,
                window_mode=CONTEXT_WINDOW_DAYS is not None,
                context_window_days=CONTEXT_WINDOW_DAYS,
                max_context_len=MAX_CONTEXT_LEN,
                band_weights=BAND_WEIGHTS,
                dt_weight_tau=DT_WEIGHT_TAU,
                ema=ema,
            )
        else:
            #train_DDP_loderunner_epoch(
            train_DDP_scalar_temporal_loderunner_epoch_9band(
                training_data=train_dataloader,
                validation_data=val_dataloader,
                num_train_batches=train_batches,
                num_val_batches=val_batches,
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                LRsched=LRsched,
                epochIDX=epochIDX,
                train_per_val=train_per_val,
                train_rcrd_filename=trn_rcrd_filename,
                val_rcrd_filename=val_rcrd_filename,
                device=device,
                rank=rank,
                world_size=world_size,
                band_weights=BAND_WEIGHTS,
                ema=ema,
            )

        print(f"[rank {rank}] finished epoch", flush=True)


        if TIME_EPOCH:
            # Synchronize before stopping the timer
            torch.cuda.synchronize(device)  # Ensure GPUs on each node sync
            #dist.barrier()  # Ensure that all nodes sync
            # Time each epoch and print to stdout
            endTime = time.time()

        epoch_time = (endTime - startTime) / 60

        # Print Summary Results
        if rank == 0:
            print(f"Completed epoch {epochIDX}...", flush=True)
            print(f"Epoch time (minutes): {epoch_time:.2f}", flush=True)

        # Save model and optimizer
        #chkpt_name_str = f'study{studyIDX:03d}_modelState_epoch{epochIDX:04d}.pth'
        #new_chkpt_path = os.path.join("./", chkpt_name_str)

        if rank == 0:
            chkpt_name_str = f"study{studyIDX:03d}_modelState_epoch{epochIDX:04d}.pth"
            new_chkpt_path = os.path.join("./", chkpt_name_str)

            print(f"Saving checkpoint: {new_chkpt_path}", flush=True)

            torch.save(
                {
                    "epoch": epochIDX,
                    "model_class": "ScalarTemporalConditionedLodeRunner_9band",
                    "backbone_class": "LodeRunner",
                    "model_args": model_args,
                    "model_state_dict": model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "noise_scale": noise_scale,
                    "predicts_delta": False,
                    "target_type": "value_9band",
                    "context_len": CONTEXT_LEN,
                    "n_bands": N_BANDS,
                    "band_keys": list(BAND_KEYS),
                    "backbone_channels": 8,
                    "hidden": HIDDEN_CHANNELS,
                    "dt_fourier_bands": DT_FOURIER_BANDS,
                    "predict_delta": PREDICT_DELTA,
                    "trend_decay_anchor": TREND_DECAY_ANCHOR,
                    "trend_slope_k": TREND_SLOPE_K,
                    "ema_decay": EMA_DECAY,
                    "ema_state_dict": (
                        ema.state_dict() if ema is not None else None
                    ),
                    "dt_weight_tau": DT_WEIGHT_TAU,
                    "band_weights": (
                        BAND_WEIGHTS.tolist()
                        if BAND_WEIGHTS is not None
                        else None
                    ),
                    "n_rollout_steps": n_rollout_steps,
                    "context_window_days": CONTEXT_WINDOW_DAYS,
                    "max_context_len": MAX_CONTEXT_LEN,
                    "target_horizon_days": TARGET_HORIZON_DAYS,
                    "train_filelist": args.train_filelist,
                    "validation_filelist": args.validation_filelist,
                    "kn_realistic_glob": args.kn_realistic_glob,
                    "kn_dense_glob": args.kn_dense_glob,
                    "norm_stats_path": norm_stats_path,
                },
                new_chkpt_path,
            )

            print(f"Saved checkpoint: {new_chkpt_path}", flush=True)

    if rank == 0:
        #############################################
        # Continue if Necessary
        #############################################
        FINISHED_TRAINING = epochIDX + 1 > total_epochs
        if not FINISHED_TRAINING:
            new_slurm_file = continuation_setup(
                new_chkpt_path, studyIDX, last_epoch=epochIDX
            )
            os.system(f"sbatch {new_slurm_file}")

if __name__ == "__main__":
    print('running main')
    args = parser.parse_args()

    rank, world_size, local_rank, device = setup_distributed()

    main(args, rank, world_size, local_rank, device)

    cleanup_distributed()
