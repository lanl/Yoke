import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from yoke.models.vit.swin.bomberman import LodeRunner
from yoke.datasets.lsc_dataset import LSC_rho2rho_temporal_DataSet
from yoke.utils.training.epoch.loderunner import train_DDP_loderunner_epoch
from yoke.utils.training.epoch.loderunner import train_DDP_loderunner_epoch_seq_context
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


class Kilonova_lc_img_DataSet(Dataset):
    def __init__(self, half_image=False, N_imgs=0):
        file_prefix_list = sorted(
            glob.glob("/net/sescratch1/atoivonen/data/KN_lightcurves/uniform_dataset_20000/lc_*.npz")
        )

        if N_imgs == 0:
            self.file_prefix_list = file_prefix_list
        else:
            self.file_prefix_list = list(np.random.choice(file_prefix_list, N_imgs, replace=False))

        random.shuffle(self.file_prefix_list)

        #self.max_timeIDX_offset = max_timeIDX_offset
        self.half_image = half_image

        # Build a global index: one entry per usable (file, startIDX)
        self.samples = []
        seqLen = 1

        for file_idx, fn in enumerate(self.file_prefix_list):
            data = np.load(fn, allow_pickle=True)
            mjd = data["arr_ztfg"][:, 0]
            n_times = len(mjd)
            data.close()

            max_start = n_times - seqLen - 1
            for startIDX in range(max_start + 1):
                self.samples.append((file_idx, startIDX))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        file_idx, startIDX = self.samples[index]
        fn = self.file_prefix_list[file_idx]

        data = np.load(fn, allow_pickle=True)

        mjd = data["arr_ztfg"][:, 0]
        t0 = mjd.min()
        t_obs = mjd - t0
        g_mag = data["arr_ztfg"][:, 1]

        seqLen = 1
        endIDX = startIDX + seqLen

        start_mag = g_mag[startIDX]
        end_mag = g_mag[endIDX]
        start_t = t_obs[startIDX]
        end_t = t_obs[endIDX]

        Dt = torch.tensor(end_t - start_t, dtype=torch.float32)

        H, W = 1120, 400

        s = torch.tensor(start_mag, dtype=torch.float32)
        start_img = s.view(1, 1, 1).expand(8, H, W)

        e = torch.tensor(end_mag, dtype=torch.float32)
        end_img = e.view(1, 1, 1).expand(8, H, W)

        data.close()
        return start_img, end_img, Dt


class Kilonova_lc_img_DataSet_seq(Dataset):
    def __init__(self, half_image=False, N_imgs=0):
        file_prefix_list = sorted(
            glob.glob("/net/sescratch1/atoivonen/data/KN_lightcurves/uniform_dataset_20000/lc_*.npz")
        )

        if N_imgs == 0:
            self.file_prefix_list = file_prefix_list
        else:
            self.file_prefix_list = list(np.random.choice(file_prefix_list, N_imgs, replace=False))

        random.shuffle(self.file_prefix_list)

        #self.max_timeIDX_offset = max_timeIDX_offset
        self.half_image = half_image

        # Build a global index: one entry per usable (file, startIDX)
        self.samples = []
        seqLen = 3

        for file_idx, fn in enumerate(self.file_prefix_list):
            data = np.load(fn, allow_pickle=True)
            mjd = data["arr_ztfg"][:, 0]
            n_times = len(mjd)
            data.close()

            max_start = n_times - seqLen - 1
            for startIDX in range(max_start + 1):
                self.samples.append((file_idx, startIDX))

    def __len__(self):
        return len(self.samples)

    
    def __getitem__(self, index):
        file_idx, startIDX = self.samples[index]
        fn = self.file_prefix_list[file_idx]

        frames = []
        seqLen = 3
        H, W = 1120, 400

        data = np.load(fn, allow_pickle=True)

        mjd = data["arr_ztfg"][:, 0]
        t0 = mjd.min()
        t_obs = mjd - t0
        g_mag = data["arr_ztfg"][:, 1]

        endIDX = startIDX + seqLen

        for i in range(seqLen):
            seq_mag = g_mag[startIDX + i]
            s = torch.tensor(seq_mag, dtype=torch.float32)
            seq_img = s.view(1, 1, 1).expand(8, H, W)
            frames.append(seq_img)

        end_mag = g_mag[endIDX]
        e = torch.tensor(end_mag, dtype=torch.float32)
        end_img = e.view(1, 1, 1).expand(8, H, W)
        frames.append(end_img)

        start_t = t_obs[startIDX]
        end_t = t_obs[endIDX]
        Dt = torch.tensor(end_t - start_t, dtype=torch.float32)

        data.close()

        img_seq = torch.stack(frames, dim=0)
        return img_seq, Dt


class Kilonova_lc_img_DataSet_channels_context(Dataset):
    def __init__(
        self,
        half_image=False,
        N_imgs=0,
        context_len=3,
        H=1120,
        W=400,
        n_channels=8,
    ):
        assert context_len <= n_channels

        file_prefix_list = sorted(
            glob.glob("/net/sescratch1/atoivonen/data/KN_lightcurves/uniform_dataset_20000/lc_*.npz")
        )

        if N_imgs == 0:
            self.file_prefix_list = file_prefix_list
        else:
            self.file_prefix_list = list(np.random.choice(file_prefix_list, N_imgs, replace=False))

        random.shuffle(self.file_prefix_list)

        self.context_len = context_len
        self.H = H
        self.W = W
        self.n_channels = n_channels
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
        g_mag = arr[:, 1]

        t0 = mjd.min()
        t_obs = mjd - t0

        target_idx = startIDX + self.context_len

        '''
        # Input: [8, H, W], where channels encode previous timesteps.
        # Right-align context. Unused earlier channels repeat earliest value.
        context_img = torch.empty(self.n_channels, self.H, self.W, dtype=torch.float32)

        earliest_mag = float(g_mag[startIDX])
        context_img[:] = earliest_mag

        offset = self.n_channels - self.context_len
        for i in range(self.context_len):
            ch = offset + i
            context_img[ch] = float(g_mag[startIDX + i])

        # Target: next scalar copied across all 8 channels.
        target_mag = float(g_mag[target_idx])
        target_img = torch.empty(self.n_channels, self.H, self.W, dtype=torch.float32)
        target_img[:] = target_mag
        '''

        context_vals = torch.empty(self.n_channels, dtype=torch.float32)

        earliest_mag = float(g_mag[startIDX])
        context_vals[:] = earliest_mag

        offset = self.n_channels - self.context_len
        for i in range(self.context_len):
            ch = offset + i
            context_vals[ch] = float(g_mag[startIDX + i])

        # expand() better for memory
        context_img = context_vals.view(self.n_channels, 1, 1).expand(
            self.n_channels,
            self.H,
            self.W,
        )

        target_mag = float(g_mag[target_idx])
        target_val = torch.tensor(target_mag, dtype=torch.float32)

        target_img = target_val.view(1, 1, 1).expand(
            self.n_channels,
            self.H,
            self.W,
        )

        Dt = torch.tensor(
            t_obs[target_idx] - t_obs[target_idx - 1],
            dtype=torch.float32,
        )

        data.close()

        return context_img, target_img, Dt


class ChannelStackAdapter(nn.Module):
    """
    Converts a sequence [B, K, C, H, W] into a fused image [B, C, H, W].
    """

    def __init__(self, in_channels: int, context_len: int, hidden_channels: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.context_len = context_len
        stacked_channels = in_channels * context_len

        self.adapter = nn.Sequential(
            nn.Conv2d(stacked_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1),
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        x_seq: [B, K, C, H, W]
        returns: [B, C, H, W]
        """
        if x_seq.ndim != 5:
            raise ValueError(f"Expected x_seq to have shape [B, K, C, H, W], got {x_seq.shape}")

        B, K, C, H, W = x_seq.shape
        if C != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {C}")
        if K != self.context_len:
            raise ValueError(f"Expected context_len={self.context_len}, got K={K}")

        x = x_seq.reshape(B, K * C, H, W)
        return self.adapter(x)


class TemporalLodeRunner(nn.Module):
    """
    Wraps a pretrained one-step LodeRunner with a temporal adapter.

    Input:
        x_seq: [B, K, C, H, W]
        in_vars, out_vars, Dt: same as original LodeRunner API

    Output:
        pred: [B, C, H, W]
    """

    def __init__(
        self,
        backbone: nn.Module,
        in_channels: int = 8,
        context_len: int = 3,
        hidden_channels: int = 64,
    ):
        super().__init__()
        self.backbone = backbone
        self.temporal_adapter = ChannelStackAdapter(
            in_channels=in_channels,
            context_len=context_len,
            hidden_channels=hidden_channels,
        )

    def forward(
        self,
        x_seq: torch.Tensor,
        in_vars: torch.Tensor,
        out_vars: torch.Tensor,
        Dt: torch.Tensor,
    ) -> torch.Tensor:
        fused_x = self.temporal_adapter(x_seq)          # [B, C, H, W]
        pred = self.backbone(fused_x, in_vars, out_vars, Dt)
        return pred


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

    model = LodeRunner(**saved_model_args)
    model.to(device)

    state_dict = checkpoint_data["model_state_dict"]

    if all(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)

    noise_scale = checkpoint_data.get("noise_scale", 0.0)
    model.noise_scale = noise_scale

    optimizer = torch.optim.AdamW(
        model.parameters(),
        **optimizer_kwargs,
    )

    if "optimizer_state_dict" in checkpoint_data:
        optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])

        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)

    starting_epoch = checkpoint_data["epoch"]

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
        "lr": 1e-5,
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

        # End-to-end fine-tuning: train adapter + backbone
        for p in model.parameters():
            p.requires_grad = True

        optimizer = torch.optim.AdamW(
            model.parameters(),
            **optimizer_kwargs,
        )

    loss_fn = nn.MSELoss(reduction="none")

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
    train_dataset = Kilonova_lc_img_DataSet_seq(
        half_image=False,
    )
    val_dataset = Kilonova_lc_img_DataSet_seq(
        half_image=False,
    )
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


        train_DDP_loderunner_epoch(
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
            seq=False,
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
                    "model_class": "LodeRunner",
                    "model_args": model_args,
                    "model_state_dict": model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "noise_scale": noise_scale,
                },
                new_chkpt_path,
            )

            '''
            save_model_and_optimizer(
                model.module,
                optimizer,
                epochIDX,
                new_chkpt_path,
                model_class=LodeRunner,
                model_args=model_args,
            )
            '''

            print(f"Saved checkpoint: {new_chkpt_path}", flush=True)

        '''
        if rank == 0:
            chkpt_name_str = f"study{studyIDX:03d}_modelState_epoch{epochIDX:04d}.pth"
            new_chkpt_path = os.path.join("./", chkpt_name_str)

            #save_model_and_optimizer(
            #    model,
            #    optimizer,
            #    epochIDX,
            #    new_chkpt_path,
            #    model_class=LodeRunner,
            #    model_args=model_args,
            #)

            print(f"Saved checkpoint: {new_chkpt_path}", flush=True)
        '''
    '''
        save_model_and_optimizer(
            model,
            optimizer,
            epochIDX,
            new_chkpt_path,
            model_class=LodeRunner,
            model_args=model_args,
        )
    '''
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
