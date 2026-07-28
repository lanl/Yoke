import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR

from yoke.models.vit.swin.bomberman import LodeRunner
from yoke.datasets.lsc_dataset import LSC_rho2rho_temporal_DataSet
from yoke.utils.training.epoch.loderunner import train_DDP_scalar_temporal_loderunner_epoch_gri
from yoke.utils.restart import continuation_setup
from yoke.utils.dataload import make_distributed_dataloader
from yoke.utils.checkpointing import load_model_and_optimizer
from yoke.utils.checkpointing import save_model_and_optimizer
from yoke.lr_schedulers import CosineWithWarmupScheduler
from yoke.helpers import cli

# FIXME remove if restructure
from torch.utils.data import Dataset, DataLoader, random_split
import glob
import random

#MEAN = 24.694652705328807
#STD = 4.67030961432848

GLOBAL_GMAG_MEAN = 24.694652705328807
GLOBAL_GMAG_STD  = 4.67030961432848
EPS = 1e-6

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

# Change some default filepaths.
parser.set_defaults(
    train_filelist="lsc240420_prefixes_train_80pct.txt",
    validation_filelist="lsc240420_prefixes_validation_10pct.txt",
    test_filelist="lsc240420_prefixes_test_10pct.txt",
)


def compute_band_normalization(
    file_prefix_list,
    band_keys=("arr_ztfg", "arr_ztfr", "arr_ztfi"),
    value_col=1,
    stats_path="kilonova_gri_norm_stats.npz",
):
    """
    Compute global per-band mean/std over the training files only.

    Saves:
        means: shape [3]
        stds:  shape [3]
    """

    sums = np.zeros(len(band_keys), dtype=np.float64)
    sums_sq = np.zeros(len(band_keys), dtype=np.float64)
    counts = np.zeros(len(band_keys), dtype=np.float64)

    for fn in file_prefix_list:
        data = np.load(fn, allow_pickle=True)

        for b, key in enumerate(band_keys):
            vals = data[key][:, value_col].astype(np.float64)

            finite = np.isfinite(vals)
            vals = vals[finite]

            sums[b] += vals.sum()
            sums_sq[b] += np.square(vals).sum()
            counts[b] += vals.size

        data.close()

    means = sums / counts
    variances = sums_sq / counts - means**2
    variances = np.maximum(variances, 1e-12)
    stds = np.sqrt(variances)

    means = means.astype(np.float32)
    stds = stds.astype(np.float32)

    np.savez(
        stats_path,
        means=means,
        stds=stds,
        band_keys=np.array(band_keys),
        value_col=value_col,
    )

    print("Saved normalization stats:", stats_path)
    print("means:", means)
    print("stds:", stds)

    return means, stds


def load_or_compute_band_normalization(
    stats_path="kilonova_gri_norm_stats.npz",
    band_keys=("arr_ztfg", "arr_ztfr", "arr_ztfi"),
    value_col=1,
):
    file_prefix_list = sorted(
        glob.glob(
            "/net/sescratch1/atoivonen/data/KN_lightcurves/uniform_dataset_20000/lc_*.npz"
        )
    )

    if os.path.exists(stats_path):
        stats = np.load(stats_path, allow_pickle=True)
        means = stats["means"].astype(np.float32)
        stds = stats["stds"].astype(np.float32)
        stats.close()

        print("Loaded normalization stats:", stats_path)
        print("means:", means)
        print("stds:", stds)

        return means, stds

    return compute_band_normalization(
        file_prefix_list=file_prefix_list,
        band_keys=band_keys,
        value_col=value_col,
        stats_path=stats_path,
    )


class Kilonova_lc_scalar_context_DataSet_gri(Dataset):
    def __init__(
        self,
        N_imgs=0,
        context_len=5,
        band_keys=("arr_ztfg", "arr_ztfr", "arr_ztfi"),
        value_col=1,
        means=None,
        stds=None,
        predicts_delta=True,
    ):
        file_prefix_list = sorted(
            glob.glob(
                "/net/sescratch1/atoivonen/data/KN_lightcurves/"
                "uniform_dataset_20000/lc_*.npz"
            )
        )

        if N_imgs == 0:
            self.file_prefix_list = file_prefix_list
        else:
            self.file_prefix_list = list(
                np.random.choice(file_prefix_list, N_imgs, replace=False)
            )

        random.shuffle(self.file_prefix_list)

        self.context_len = context_len
        self.band_keys = tuple(band_keys)
        self.value_col = value_col
        self.n_channels = len(self.band_keys)
        self.predicts_delta = predicts_delta

        if means is None:
            raise ValueError(
                "means must be provided for per-band normalization. "
                "Expected shape [n_channels], e.g. [g_mean, r_mean, i_mean]."
            )

        if stds is None:
            raise ValueError(
                "stds must be provided for per-band normalization. "
                "Expected shape [n_channels], e.g. [g_std, r_std, i_std]."
            )

        self.means = np.asarray(means, dtype=np.float32)
        self.stds = np.asarray(stds, dtype=np.float32)

        if self.means.shape[0] != self.n_channels:
            raise ValueError(
                f"means has length {self.means.shape[0]}, "
                f"but n_channels={self.n_channels}"
            )

        if self.stds.shape[0] != self.n_channels:
            raise ValueError(
                f"stds has length {self.stds.shape[0]}, "
                f"but n_channels={self.n_channels}"
            )

        if np.any(self.stds <= 0):
            raise ValueError(f"All stds must be positive. Got stds={self.stds}")

        self.samples = []

        for file_idx, fn in enumerate(self.file_prefix_list):
            data = np.load(fn, allow_pickle=True)

            # Assume all bands share the same time grid.
            mjd = data[self.band_keys[0]][:, 0]
            n_times = len(mjd)

            data.close()

            max_start = n_times - context_len - 1
            for startIDX in range(max_start + 1):
                self.samples.append((file_idx, startIDX))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        file_idx, startIDX = self.samples[index]
        fn = self.file_prefix_list[file_idx]

        data = np.load(fn, allow_pickle=True)

        # Time grid from the first band.
        # Assumes arr_ztfg, arr_ztfr, arr_ztfi have matching MJD columns.
        mjd = data[self.band_keys[0]][:, 0].astype(np.float32)

        # Stack one scalar value column from each band.
        # vals shape: [T, 3] for g/r/i.
        vals = np.stack(
            [
                data[key][:, self.value_col].astype(np.float32)
                for key in self.band_keys
            ],
            axis=1,
        )

        data.close()

        # Per-band normalization.
        # vals[:, 0] = normalized g
        # vals[:, 1] = normalized r
        # vals[:, 2] = normalized i
        vals = (vals - self.means[None, :]) / (self.stds[None, :] + EPS)

        t0 = mjd.min()
        t_obs = mjd - t0

        target_idx = startIDX + self.context_len

        # Context values shape: [context_len, 3]
        context_vals = vals[startIDX:target_idx]

        # Relative observation times shape: [context_len]
        rel_t = t_obs[startIDX:target_idx] - t_obs[startIDX]
        rel_t = rel_t.astype(np.float32)

        # Flatten context as:
        # [g0, r0, i0, g1, r1, i1, ..., gK, rK, iK]
        context_flat = context_vals.reshape(-1).astype(np.float32)

        # Final input shape:
        # [(3 * context_len) + context_len]
        #
        # For context_len=5:
        # x.shape == [20]
        x = np.concatenate([context_flat, rel_t], axis=0)
        x = torch.tensor(x, dtype=torch.float32)

        if self.predicts_delta:
            # Predict normalized delta for each band:
            # [delta_g, delta_r, delta_i]
            target_vals = vals[target_idx] - vals[target_idx - 1]
        else:
            # Predict normalized absolute next value:
            # [g_next, r_next, i_next]
            target_vals = vals[target_idx]

        target = torch.tensor(target_vals, dtype=torch.float32)

        Dt = torch.tensor(
            t_obs[target_idx] - t_obs[target_idx - 1],
            dtype=torch.float32,
        )

        return x, target, Dt


class Kilonova_lc_scalar_context_DataSet(Dataset):
    def __init__(
        self,
        N_imgs=0,
        context_len=5,
    ):
        file_prefix_list = sorted(
            glob.glob("/net/sescratch1/atoivonen/data/KN_lightcurves/uniform_dataset_20000/lc_*.npz")
        )

        if N_imgs == 0:
            self.file_prefix_list = file_prefix_list
        else:
            self.file_prefix_list = list(np.random.choice(file_prefix_list, N_imgs, replace=False))

        random.shuffle(self.file_prefix_list)

        self.context_len = context_len
        self.samples = []

        for file_idx, fn in enumerate(self.file_prefix_list):
            data = np.load(fn, allow_pickle=True)
            mjd = data["arr_ztfg"][:, 0]
            n_times = len(mjd)
            data.close()

            max_start = n_times - context_len - 1
            for startIDX in range(max_start + 1):
                self.samples.append((file_idx, startIDX))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        file_idx, startIDX = self.samples[index]
        fn = self.file_prefix_list[file_idx]

        data = np.load(fn, allow_pickle=True)
        arr = data["arr_ztfg"]

        mjd = arr[:, 0]
        g_mag = arr[:, 1].astype(np.float32)

        g_mag = (g_mag - GLOBAL_GMAG_MEAN) / (GLOBAL_GMAG_STD + EPS)

        t0 = mjd.min()
        t_obs = mjd - t0

        target_idx = startIDX + self.context_len

        mags = g_mag[startIDX:target_idx].astype(np.float32)

        # context_len values; first dt is 0, remaining are relative times
        rel_t = t_obs[startIDX:target_idx] - t_obs[startIDX]
        rel_t = rel_t.astype(np.float32)

        x = np.concatenate([mags, rel_t], axis=0)
        x = torch.tensor(x, dtype=torch.float32)

        delta_mag = g_mag[target_idx] - g_mag[target_idx - 1]
        target = torch.tensor(delta_mag, dtype=torch.float32)

        Dt = torch.tensor(
            t_obs[target_idx] - t_obs[target_idx - 1],
            dtype=torch.float32,
        )

        # Keep target image-shaped only if your datastep still expects image targets.
        # Better is to update datastep to use scalar target directly.
        data.close()
        return x, target, Dt


class ScalarTemporalConditionedLodeRunner_gri(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        context_len: int = 5,
        n_input_channels: int = 3,
        n_output_channels: int = 3,
        image_size=(1120, 400),
        backbone_channels: int = 8,
        hidden: int = 64,
    ):
        super().__init__()

        self.backbone = backbone
        self.context_len = context_len
        self.n_input_channels = n_input_channels
        self.n_output_channels = n_output_channels
        self.image_size = image_size
        self.backbone_channels = backbone_channels

        # Dataset x layout:
        #   [g0, r0, i0, g1, r1, i1, ..., gK, rK, iK, t0, t1, ..., tK]
        #
        # input_dim = context_len * n_input_channels + context_len
        input_dim = context_len * n_input_channels + context_len

        # Maps scalar temporal context into the 8 pseudo-channels expected by
        # the pretrained LodeRunner backbone.
        self.conditioner = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, backbone_channels),
        )

        # Maps the 8-channel LodeRunner output back to 3 scalar predictions:
        #   [delta_g, delta_r, delta_i]
        self.output_head = nn.Sequential(
            nn.Linear(backbone_channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_output_channels),
        )

    def forward(self, x, in_vars, out_vars, Dt):
        """
        x: [B, context_len * n_input_channels + context_len]

        For 3-band, context_len=5:
            x.shape == [B, 20]

        Returns:
            pred: [B, 3]
        """
        B = x.shape[0]
        H, W = self.image_size

        channel_vals = self.conditioner(x)  # [B, 8]

        pseudo_img = channel_vals.view(
            B,
            self.backbone_channels,
            1,
            1,
        ).expand(
            B,
            self.backbone_channels,
            H,
            W,
        )

        backbone_in_vars = torch.arange(self.backbone_channels, device=x.device)
        backbone_out_vars = torch.arange(self.backbone_channels, device=x.device)

        pred_img = self.backbone(
            pseudo_img,
            backbone_in_vars,
            backbone_out_vars,
            Dt,
        )  # [B, 8, H, W]

        # Collapse spatial dimensions to 8 backbone-channel summaries.
        pred_channel_vals = pred_img.mean(dim=(2, 3))  # [B, 8]

        # Convert 8 backbone channels to 3 output bands.
        pred = self.output_head(pred_channel_vals)  # [B, 3]

        return pred


class ScalarTemporalConditionedLodeRunner(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        context_len: int = 5,
        image_size=(1120, 400),
        n_channels: int = 8,
        hidden: int = 64,
    ):
        super().__init__()
        self.backbone = backbone
        self.context_len = context_len
        self.image_size = image_size
        self.n_channels = n_channels

        # mags + dts
        self.conditioner = nn.Sequential(
            nn.Linear(2 * context_len, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_channels),
        )

    def forward(self, x, in_vars, out_vars, Dt):
        """
        x: [B, 2 * context_len]
           first context_len entries are magnitudes
           second context_len entries are temporal deltas
        """
        B = x.shape[0]
        H, W = self.image_size

        channel_vals = self.conditioner(x)  # [B, 8]

        pseudo_img = channel_vals.view(B, self.n_channels, 1, 1).expand(
            B,
            self.n_channels,
            H,
            W,
        )

        pred_img = self.backbone(pseudo_img, in_vars, out_vars, Dt)

        # Convert LodeRunner image output back to scalar delta prediction
        pred_scalar = pred_img.mean(dim=(1, 2, 3))

        return pred_scalar


def load_direct_loderunner_checkpoint(
    checkpoint_path,
    model_args,
    optimizer_kwargs,
    device,
):
    checkpoint_data = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    saved_model_args = checkpoint_data.get("model_args", model_args)
    context_len = checkpoint_data.get("context_len", 5)

    backbone = LodeRunner(**saved_model_args).to(device)

    model = ScalarTemporalConditionedLodeRunner_gri(
        backbone=backbone,
        context_len=context_len,
        n_input_channels=checkpoint_data.get("n_input_channels", 3),
        n_output_channels=checkpoint_data.get("n_output_channels", 3),
        image_size=saved_model_args["image_size"],
        backbone_channels=checkpoint_data.get("backbone_channels", 8),
        hidden=checkpoint_data.get("hidden", 64),
    ).to(device)

    state_dict = checkpoint_data["model_state_dict"]

    # Remove DDP prefix if present
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {
            k.replace("module.", "", 1): v
            for k, v in state_dict.items()
        }

    # Detect checkpoint type
    is_wrapper_checkpoint = any(
        k.startswith("backbone.") for k in state_dict.keys()
    )

    # -------------------------------------------------
    # OLD plain LodeRunner checkpoint
    # -------------------------------------------------
    if not is_wrapper_checkpoint:

        missing_keys, unexpected_keys = model.backbone.load_state_dict(
            state_dict,
            strict=False,
        )

        print("Loaded old LodeRunner checkpoint into model.backbone")
        print("Missing backbone keys:", missing_keys)
        print("Unexpected backbone keys:", unexpected_keys)

        # This is NOT a true continuation.
        # Conditioner is newly initialized.
        starting_epoch = 0

    # -------------------------------------------------
    # NEW ScalarTemporalConditionedLodeRunner checkpoint
    # -------------------------------------------------
    else:

        model.load_state_dict(state_dict, strict=True)

        print("Loaded ScalarTemporalConditionedLodeRunner checkpoint")

        starting_epoch = checkpoint_data.get("epoch", 0)

    noise_scale = checkpoint_data.get("noise_scale", 0.0)
    model.backbone.noise_scale = noise_scale

    # Freeze pretrained backbone
    for p in model.backbone.parameters():
        p.requires_grad = False

    # Train conditioner
    for p in model.conditioner.parameters():
        p.requires_grad = True

    #optimizer = torch.optim.AdamW(
    #    model.conditioner.parameters(),
    #    **optimizer_kwargs,
    #)
    
    optimizer = torch.optim.AdamW(
        list(model.conditioner.parameters()) +
        list(model.output_head.parameters()),
        **optimizer_kwargs,
    )

    # Only restore optimizer for TRUE continuation checkpoints
    if (
        is_wrapper_checkpoint
        and "optimizer_state_dict" in checkpoint_data
    ):

        optimizer.load_state_dict(
            checkpoint_data["optimizer_state_dict"]
        )

        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)

    return model, optimizer, starting_epoch


def setup_distributed():
    # ----- 1) Basic setup & environment variables -----
    # Rely on Slurm variables: SLURM_PROCID, SLURM_NTASKS, SLURM_LOCALID, etc.
    rank = int(os.environ["SLURM_PROCID"])  # global rank
    world_size = int(os.environ["SLURM_NTASKS"])  # total number of processes
    local_rank = int(os.environ["SLURM_LOCALID"])  # local rank (GPU index on this node)

    master_addr = os.environ["MASTER_ADDR"]
    master_port = os.environ["MASTER_PORT"]

    # ----- 2) Set the current GPU device for this process -----
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # ----- 3) Initialize the process group -----
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        world_size=world_size,
        rank=rank,
    )

    return rank, world_size, local_rank, device


def cleanup_distributed():
    # ----- 8) Clean up (optional) -----
    dist.destroy_process_group()


def main(args, rank, world_size, local_rank, device):
    #############################################
    # Process Inputs
    #############################################
    # Study ID
    studyIDX = args.studyIDX

    # Resources
    Ngpus = args.Ngpus
    Knodes = args.Knodes

    # Data Paths
    train_filelist = args.FILELIST_DIR + args.train_filelist
    validation_filelist = args.FILELIST_DIR + args.validation_filelist

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

    optimizer_kwargs = {
        "lr": 1e-4,# 1e-4, #1e-5
        "betas": (0.9, 0.999),
        "eps": 1e-08,
        "weight_decay": 0.01,
    }


    if CONTINUATION:
        model, optimizer, starting_epoch = load_direct_loderunner_checkpoint(
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

            model.noise_scale = noise_scale

            backbone = model

            model = ScalarTemporalConditionedLodeRunner_gri(
                backbone=backbone,
                context_len=CONTEXT_LEN,
                n_input_channels=3,
                n_output_channels=3,
                image_size=model_args["image_size"],
                backbone_channels=8,
                hidden=HIDDEN_CHANNELS,
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

    BAND_KEYS = ("arr_ztfg", "arr_ztfr", "arr_ztfi")
    VALUE_COL = 1
    N_BANDS = len(BAND_KEYS)

    norm_stats_path = "kilonova_gri_norm_stats.npz"

    if rank == 0:
        band_means, band_stds = load_or_compute_band_normalization(
            stats_path=norm_stats_path,
            band_keys=BAND_KEYS,
            value_col=VALUE_COL,
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

    train_dataset = Kilonova_lc_scalar_context_DataSet_gri(
        context_len=CONTEXT_LEN,
        band_keys=BAND_KEYS,
        value_col=VALUE_COL,
        means=band_means,
        stds=band_stds,
    )

    val_dataset = Kilonova_lc_scalar_context_DataSet_gri(
        context_len=CONTEXT_LEN,
        band_keys=BAND_KEYS,
        value_col=VALUE_COL,
        means=band_means,
        stds=band_stds,
    )


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


        #train_DDP_loderunner_epoch(
        train_DDP_scalar_temporal_loderunner_epoch_gri(
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
                    "model_class": "ScalarTemporalConditionedLodeRunner_gri",
                    "backbone_class": "LodeRunner",
                    "model_args": model_args,
                    "model_state_dict": model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "noise_scale": noise_scale,
                    "predicts_delta": True,
                    "target_type": "delta_gri",
                    "context_len": CONTEXT_LEN,
                    "n_input_channels": 3,
                    "n_output_channels": 3,
                    "backbone_channels": 8,
                    "hidden": 64,
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
