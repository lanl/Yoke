"""Module for BomberMan network structures.

nn.Module allowing processing of variable channel image input through a SWIN-V2
U-Net architecture then re-embedded as a variable channel image.

This network architecture will serve as the foundation for a hydro-code
emulator.

"""

from collections.abc import Callable, Iterable
import math
import random

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler
from lightning.pytorch import LightningModule

from yoke.models.vit.swin.unet import SwinUnetBackbone
from yoke.models.vit.patch_embed import ParallelVarPatchEmbed
from yoke.models.vit.patch_manipulation import Unpatchify
from yoke.models.vit.aggregate_variables import AggVars
from yoke.models.vit.embedding_encoders import (
    VarEmbed,
    PosEmbed,
    TimeEmbed,
)
from yoke.lr_schedulers import CosineWithWarmupScheduler
from yoke.helpers.training_design import validate_patch_and_window


class LodeRunner(nn.Module):
    """LodeRunner neural network.

    Parallel-patch embedding with SWIN U-Net backbone and
    unpatchification. This module will take in a variable-channel image format
    and output an equivalent variable-channel image formate. This will serves
    as a prototype foundational architecture for multi-material, multi-physics,
    surrogate models.

    Args:
        default_vars (list[str]): List of default variables to be used for training
        image_size (tuple[int, int]): Height and width, in pixels, of input image.
        patch_size (tuple[int, int]): Height and width pixel dimensions of patch in
                                      initial embedding.
        emb_dim (int): Initial embedding dimension.
        emb_factor (int): Scale of embedding in each patch merge/expand.
        num_heads (int): Number of heads in the MSA layers.
        block_structure (int, int, int, int): Tuple specifying the number of SWIN
                                              encoders in each block structure
                                              separated by the patch-merge layers.
        window_sizes (list(4*(int, int))): Window sizes within each SWIN encoder/decoder.
        patch_merge_scales (list(3*(int, int))): Height and width scales used in
                                                 each patch-merge layer.
        verbose (bool): When TRUE, windowing and merging dimensions are printed
                        during initialization.

    """

    def __init__(
        self,
        default_vars: list[str],
        image_size: Iterable[int, int] = (1120, 800),
        patch_size: Iterable[int, int] = (10, 10),
        embed_dim: int = 128,
        emb_factor: int = 2,
        num_heads: int = 8,
        block_structure: Iterable[int, int, int, int] = (1, 1, 3, 1),
        window_sizes: Iterable[(int, int), (int, int), (int, int), (int, int)] = [
            (8, 8),
            (8, 8),
            (4, 4),
            (2, 2),
        ],
        patch_merge_scales: Iterable[(int, int), (int, int), (int, int)] = [
            (2, 2),
            (2, 2),
            (2, 2),
        ],
        noise_scale: float = 0.0,
        verbose: bool = False,
    ) -> None:
        """Initialization for class."""
        super().__init__()

        self.default_vars = default_vars
        self.max_vars = len(self.default_vars)
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.emb_factor = emb_factor
        self.num_heads = num_heads
        self.block_structure = block_structure
        self.window_sizes = window_sizes
        self.patch_merge_scales = patch_merge_scales
        self.noise_scale = noise_scale

        # Validate patch_size, window_sizes, and patch_merge_scales before proceeding.
        valid = validate_patch_and_window(
            image_size=image_size,
            patch_size=patch_size,
            window_sizes=window_sizes,
            patch_merge_scales=patch_merge_scales,
        )
        assert np.all(valid), (
            "Invalid combination of image_size, patch_size, window_sizes, "
            "and patch_merge_scales!"
        )

        # First embed the image as a sequence of tokenized patches. Each
        # channel is embedded independently.
        self.parallel_embed = ParallelVarPatchEmbed(
            max_vars=self.max_vars,
            img_size=self.image_size,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            norm_layer=None,
        )

        # Encode tokens corresponding to each variable with a learnable tag
        self.var_embed_layer = VarEmbed(self.default_vars, self.embed_dim)

        # Aggregate variable tokenizations using an attention mechanism
        self.agg_vars = AggVars(self.embed_dim, self.num_heads)

        # Encode each patch with position information. Position encoding is
        # only index-aware and does not take into account actual spatial
        # information.
        self.pos_embed = PosEmbed(
            self.embed_dim,
            self.patch_size,
            self.image_size,
            self.parallel_embed.num_patches,
        )

        # Encode temporal-offset information using a linear mapping.
        self.temporal_encoding = TimeEmbed(self.embed_dim)

        # Pass encoded patch tokens through a SWIN-Unet structure
        self.unet = SwinUnetBackbone(
            emb_size=self.embed_dim,
            emb_factor=self.emb_factor,
            patch_grid_size=self.parallel_embed.grid_size,
            block_structure=self.block_structure,
            num_heads=self.num_heads,
            window_sizes=self.window_sizes,
            patch_merge_scales=self.patch_merge_scales,
            verbose=verbose,
        )

        # Linear embed the last dimension into V*p_h*p_w
        self.linear4unpatch = nn.Linear(
            self.embed_dim, self.max_vars * self.patch_size[0] * self.patch_size[1]
        )

        # Unmap the tokenized embeddings to variables and images.
        self.unpatch = Unpatchify(
            total_num_vars=self.max_vars,
            patch_grid_size=self.parallel_embed.grid_size,
            patch_size=self.patch_size,
        )

    def forward(
        self,
        x: torch.Tensor,
        in_vars: torch.Tensor,
        out_vars: torch.Tensor,
        lead_times: torch.Tensor,
    ) -> torch.Tensor:
        """Forward method for LodeRunner."""
        # WARNING!: Most likely the `in_vars` and `out_vars` need to be tensors
        # of integers corresponding to variables in the `default_vars` list.

        # Noise injection:
        l2_norm = torch.sqrt((x * x).sum(dim=(1, 2, 3), keepdim=True))
        noise = torch.randn_like(x)
        x = x + self.noise_scale * l2_norm * noise

        # Embed input
        # varIDXs = self.var_embed_layer.get_var_ids(tuple(in_vars), x.device)
        x = self.parallel_embed(x, in_vars)

        # Encode variables
        x = self.var_embed_layer(x, in_vars)

        # Aggregate variables
        x = self.agg_vars(x)

        # Encode patch positions, spatial information
        x = self.pos_embed(x)

        # Encode temporal information
        x = self.temporal_encoding(x, lead_times)

        # Pass through SWIN-V2 U-Net encoder
        x = self.unet(x)

        # Use linear map to remap to correct variable and patchsize dimension
        x = self.linear4unpatch(x)

        # Unpatchify back to original shape
        x = self.unpatch(x)

        # Select only entries corresponding to out_vars for loss
        # out_var_ids = self.var_embed_layer.get_var_ids(tuple(out_vars), x.device)
        preds = x[:, out_vars]

        return preds


class Lightning_LodeRunner(LightningModule):
    """Lightning wrapper for LodeRunner.

    Wrap LodeRunner torch.nn.Module class in a lightning.LightningModule for
    ease of parallelization and encapsulation of training strategy.

    Args:
        model (nn.Module): Pre-initialized nn.Module to wrap
        in_vars (torch.Tensor): Input channels to train LodeRunner on
        out_vars (torch.Tensor): Output channels to train LodeRunner on
        lr_scheduler (_LRScheduler): Learning-rate scheduler to use with optimizer
        scheduler_params (dict): Keyword arguments to initialize scheduler
        loss_fn (Callable): Loss function used to evaluate predictions at each timestep.
        scheduled_sampling_scheduler (Callable): Function that accepts the current
            training step and returns a number in [0, 1] for scheduled sampling
            probability.
    """

    def __init__(
        self,
        model: nn.Module,
        in_vars: torch.Tensor = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
        out_vars: torch.Tensor = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
        lr_scheduler: _LRScheduler = None,
        scheduler_params: dict = None,
        loss_fn: Callable = nn.MSELoss(reduction="none"),
        scheduled_sampling_scheduler: Callable = lambda global_step: 1.0,
    ) -> None:
        """Initialization for Lightning wrapper."""
        super().__init__()
        self.model = model
        self.lr_scheduler = lr_scheduler or CosineWithWarmupScheduler
        self.scheduler_params = scheduler_params or {}
        self.scheduled_sampling_scheduler = scheduled_sampling_scheduler
        self.loss_fn = loss_fn

        # Register buffers to ensure auto-transfer to devices as needed.
        self.register_buffer("in_vars", in_vars)
        self.register_buffer("out_vars", out_vars)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Setup optimizer with scheduler."""
        # Optimizer setup
        optimizer = torch.optim.AdamW(self.model.parameters())

        # Initialize LR scheduler
        scheduler = self.lr_scheduler(optimizer, **self.scheduler_params)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Step scheduler every batch.
                "frequency": 1,  # Step every batch (default for "step")
            },
        }

    def forward(self, X: torch.Tensor, lead_times: torch.Tensor) -> torch.Tensor:
        """Forward method for Lightning wrapper."""
        # Forward pass through the custom model
        return self.model(
            X, lead_times=lead_times, in_vars=self.in_vars, out_vars=self.out_vars
        )

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Execute training step."""
        # Compute forward pass, accounting for special training schemes like
        # scheduled sampling.
        img_seq, lead_times = batch  # Unpack batch
        pred_seq = []
        scheduled_prob = self.scheduled_sampling_scheduler(self.current_epoch)
        for k, k_img in enumerate(torch.unbind(img_seq[:, :-1], dim=1)):
            if k == 0:
                # Forward pass for the initial step
                pred_img = self(k_img, lead_times)
            else:
                # Apply scheduled sampling
                if random.random() < scheduled_prob:
                    current_input = k_img
                else:
                    current_input = pred_img
                pred_img = self(current_input, lead_times)

            # Store the prediction
            pred_seq.append(pred_img)

        # Combine predictions into a tensor of shape [B, SeqLength, C, H, W]
        pred_seq = torch.stack(pred_seq, dim=1)

        # Per-sample loss
        losses = self.loss_fn(pred_seq, img_seq[:, 1:])
        # self.log("train_loss_per_sample", losses, on_epoch=True, on_step=True)

        batch_loss = losses.mean()
        if hasattr(self, "trainer") and self.trainer.training:
            self.log("train_loss", batch_loss, sync_dist=True)
            self.log("scheduled_prob", scheduled_prob, sync_dist=True)

        return batch_loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """Execute validation step."""
        # Compute forward pass.
        img_seq, lead_times = batch  # Unpack batch
        pred_seq = []
        for k, k_img in enumerate(torch.unbind(img_seq[:, :-1], dim=1)):
            # For now, stick to next time step prediction for validation step.
            pred_img = self(k_img, lead_times)

            # Store the prediction
            pred_seq.append(pred_img)

        # Combine predictions into a tensor of shape [B, SeqLength, C, H, W]
        pred_seq = torch.stack(pred_seq, dim=1)

        # Per-sample loss
        losses = self.loss_fn(pred_seq, img_seq[:, 1:])
        # self.log("val_loss_per_sample", losses, on_epoch=True, on_step=True)

        batch_loss = losses.mean()
        if hasattr(self, "trainer") and self.trainer.validating:
            self.log("val_loss", batch_loss, sync_dist=True)


class ScalarTemporalConditionedLodeRunner_gri(nn.Module):
    """Scalar-temporal wrapper around a pretrained LodeRunner backbone.

    Maps a scalar temporal context (flattened per-band values plus relative
    observation times) into the pseudo-channel image expected by a pretrained
    LodeRunner backbone, then collapses the backbone's spatial output back to a
    small number of scalar band predictions. Used for the kilonova light-curve
    (g/r/i) forecasting task.

    Args:
        backbone (nn.Module): Pretrained LodeRunner backbone.
        context_len (int): Number of context timesteps.
        n_input_channels (int): Number of input bands (e.g. 3 for g/r/i).
        n_output_channels (int): Number of predicted bands.
        image_size (tuple): Spatial size (H, W) fed to the backbone.
        backbone_channels (int): Number of pseudo-channels the backbone expects.
        hidden (int): Hidden width of the conditioner/output-head MLPs.
    """

    def __init__(
        self,
        backbone: nn.Module,
        context_len: int = 5,
        n_input_channels: int = 3,
        n_output_channels: int = 3,
        image_size: tuple[int, int] = (1120, 400),
        backbone_channels: int = 8,
        hidden: int = 64,
    ) -> None:
        """Initialize conditioner and output-head around the backbone."""
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

    def forward(
        self,
        x: torch.Tensor,
        in_vars: torch.Tensor,
        out_vars: torch.Tensor,
        Dt: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Scalar temporal context of shape
                [B, context_len * n_input_channels + context_len]. For 3-band,
                context_len=5, x.shape == [B, 20].
            in_vars (torch.Tensor): Kept for LodeRunner API compatibility.
            out_vars (torch.Tensor): Kept for LodeRunner API compatibility.
            Dt (torch.Tensor): Lead-time tensor passed to the backbone.

        Returns:
            pred (torch.Tensor): Predictions of shape [B, n_output_channels].
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


class ScalarTemporalConditionedLodeRunner_9band(nn.Module):
    """Scalar-temporal wrapper for sparse, irregular 9-band light curves.

    Like ``ScalarTemporalConditionedLodeRunner_gri`` this maps a scalar temporal
    context into the pseudo-channel image expected by a pretrained LodeRunner
    backbone and collapses the backbone output back to per-band predictions.
    Unlike the g/r/i wrapper, the context is a merged event stream where each
    event carries its own band identity (via a one-hot encoding), so the model
    handles sparse and irregular sampling with missing bands. The output head
    emits a prediction for every band at the requested lead time, which supports
    forecasting all observatories' future observations from real data.

    Args:
        backbone (nn.Module): Pretrained LodeRunner backbone.
        context_len (int): Number of context events.
        n_bands (int): Number of bands (input band identities and output
            predictions), e.g. 9.
        image_size (tuple): Spatial size (H, W) fed to the backbone.
        backbone_channels (int): Number of pseudo-channels the backbone expects.
        hidden (int): Hidden width of the conditioner/output-head MLPs.
    """

    def __init__(
        self,
        backbone: nn.Module,
        context_len: int = 5,
        n_bands: int = 9,
        image_size: tuple[int, int] = (1120, 400),
        backbone_channels: int = 8,
        hidden: int = 64,
        context_window_days: float = None,
        dt_fourier_bands: int = 0,
    ) -> None:
        """Initialize conditioner and output-head around the backbone.

        Args:
            context_len (int): Number of context events per sample. In
                time-window mode this is the padded width (``max_context_len``).
            context_window_days (float): When set, the dataset selects context
                by a trailing time window and pads it with a per-event validity
                flag, so each event carries an extra ``valid`` feature and the
                per-event width is ``3 + n_bands`` instead of ``2 + n_bands``.
                When None (default), the legacy fixed-count layout is used.
            dt_fourier_bands (int): Number of Fourier (sinusoidal) frequency
                bands used to encode the lead time ``Dt`` and inject it into the
                trainable conditioner and output head. When ``0`` (default) the
                feature is DISABLED and the architecture is byte-identical to the
                legacy model: neither MLP sees ``Dt`` directly (lead time reaches
                the output only through the frozen backbone). When ``> 0``, a
                ``2 * dt_fourier_bands``-wide encoding ``[sin(Dt·f), cos(Dt·f)]``
                over a fixed log-spaced frequency bank is concatenated onto BOTH
                MLP inputs, so the trainable path can learn a smooth, nonlinear,
                per-band dependence on lead time (e.g. late-time decay) rather
                than a lead-time-independent persistence value. The frequency
                bank is a non-trainable buffer.
        """
        super().__init__()

        self.backbone = backbone
        self.context_len = context_len
        self.n_bands = n_bands
        self.image_size = image_size
        self.backbone_channels = backbone_channels
        self.context_window_days = context_window_days
        self.dt_fourier_bands = dt_fourier_bands

        # Dataset x layout, flattened per event. Fixed-count mode:
        #   [value, rel_t, one_hot_band(n_bands)] * context_len   -> 2 + n_bands
        # Time-window mode adds a validity flag so padding is carried in x:
        #   [value, rel_t, valid, one_hot_band(n_bands)] * context_len -> 3 + n_bands
        per_event_width = 3 + n_bands if context_window_days is not None else 2 + n_bands
        input_dim = context_len * per_event_width

        # Fourier lead-time encoding. When enabled, a fixed log-spaced frequency
        # bank (periods ~0.5 -> 15 days, matching the forecast horizon) turns the
        # scalar Dt into a 2*dt_fourier_bands feature vector that the trainable
        # MLPs consume directly. Registered as a buffer so it saves/loads and
        # moves with .to(device) but is excluded from the optimizer's param list.
        if dt_fourier_bands > 0:
            periods = torch.logspace(
                math.log10(0.5), math.log10(15.0), dt_fourier_bands
            )
            self.register_buffer("dt_freqs", 2.0 * math.pi / periods)
            dt_extra = 2 * dt_fourier_bands
        else:
            dt_extra = 0

        # Maps the scalar temporal event stream (plus the Fourier Dt encoding,
        # when enabled) into the pseudo-channels expected by the backbone.
        self.conditioner = nn.Sequential(
            nn.Linear(input_dim + dt_extra, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, backbone_channels),
        )

        # Maps the backbone-channel summary (plus the Fourier Dt encoding, when
        # enabled) back to one prediction per band.
        self.output_head = nn.Sequential(
            nn.Linear(backbone_channels + dt_extra, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_bands),
        )

    def _encode_dt(self, Dt: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Fourier-encode the lead time for the trainable path.

        Args:
            Dt (torch.Tensor): Lead-time tensor of shape [B] (or broadcastable).
            batch_size (int): Batch size B, used to size the disabled-path output.

        Returns:
            torch.Tensor: [B, 2 * dt_fourier_bands] of ``[sin(Dt·f), cos(Dt·f)]``
            when enabled, else an empty [B, 0] tensor (so the concat is a no-op
            and the disabled path is identical to the legacy model).
        """
        if self.dt_fourier_bands == 0:
            return Dt.new_zeros((batch_size, 0))

        # [B, 1] * [1, bands] -> [B, bands]
        angles = Dt.reshape(batch_size, 1) * self.dt_freqs.reshape(1, -1)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)

    def forward(
        self,
        x: torch.Tensor,
        in_vars: torch.Tensor,
        out_vars: torch.Tensor,
        Dt: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Merged event-stream context of shape
                [B, context_len * (2 + n_bands)].
            in_vars (torch.Tensor): Kept for LodeRunner API compatibility.
            out_vars (torch.Tensor): Kept for LodeRunner API compatibility.
            Dt (torch.Tensor): Lead-time tensor. Passed to the backbone and, when
                ``dt_fourier_bands > 0``, Fourier-encoded and concatenated onto
                both the conditioner and output-head inputs so the trainable path
                can condition directly on lead time.

        Returns:
            pred (torch.Tensor): Predictions of shape [B, n_bands].
        """
        B = x.shape[0]
        H, W = self.image_size

        # Fourier Dt encoding for the trainable path ([B, 0] when disabled, so
        # both concats below are no-ops and match the legacy architecture).
        dt_feat = self._encode_dt(Dt, B)

        channel_vals = self.conditioner(
            torch.cat([x, dt_feat], dim=1)
        )  # [B, backbone_channels]

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
        )  # [B, backbone_channels, H, W]

        # Collapse spatial dimensions to backbone-channel summaries.
        pred_channel_vals = pred_img.mean(dim=(2, 3))  # [B, backbone_channels]

        # Convert backbone channels to per-band predictions, conditioning the
        # head on lead time via the same Fourier encoding ([B, 0] when disabled).
        pred = self.output_head(
            torch.cat([pred_channel_vals, dt_feat], dim=1)
        )  # [B, n_bands]

        return pred


if __name__ == "__main__":
    from yoke.utils.parameters import count_torch_params

    device = "cuda" if torch.cuda.is_available() else "cpu"

    default_vars = [
        "cu_pressure",
        "cu_density",
        "cu_temperature",
        "al_pressure",
        "al_density",
        "al_temperature",
        "ss_pressure",
        "ss_density",
        "ss_temperature",
        "ply_pressure",
        "ply_density",
        "ply_temperature",
        "air_pressure",
        "air_density",
        "air_temperature",
        "hmx_pressure",
        "hmx_density",
        "hmx_temperature",
        "r_vel",
        "z_vel",
    ]

    # (B, C, H, W)
    x = torch.rand(5, 4, 1120, 800)
    x = x.type(torch.FloatTensor).to(device)

    lead_times = torch.rand(5).to(device)  # Lead time for each entry in batch
    # x_vars = ["cu_density", "ss_density", "ply_density", "air_density"]
    x_vars = torch.tensor([1, 7, 10, 13]).to(device)

    # out_vars = ["cu_density", "ss_density", "ply_density", "air_density"]
    out_vars = torch.tensor([1, 7, 10, 13]).to(device)

    # Common model setup for LodeRunner
    #
    # NOTE: For half-image `image_size = (1120, 400)` can just halve the second
    # patch_size dimension.
    emb_factor = 2
    patch_size = (10, 10)
    image_size = (1120, 800)
    num_heads = 8
    window_sizes = [(8, 8), (8, 8), (4, 4), (2, 2)]
    patch_merge_scales = [(2, 2), (2, 2), (2, 2)]

    # Tiny size
    embed_dim = 96
    block_structure = (1, 1, 3, 1)

    # Test LodeRunner architecture.
    lode_runner = LodeRunner(
        default_vars=default_vars,
        image_size=image_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        emb_factor=emb_factor,
        num_heads=num_heads,
        block_structure=block_structure,
        window_sizes=window_sizes,
        patch_merge_scales=patch_merge_scales,
        verbose=False,
    ).to(device)
    loderunner_out = lode_runner(x, x_vars, out_vars, lead_times)
    print("LodeRunner-tiny output shape:", loderunner_out.shape)
    print("LodeRunner-tiny output has NaNs:", torch.isnan(loderunner_out).any())
    print("LodeRunner-tiny parameters:", count_torch_params(lode_runner, trainable=True))

    # Test lightning wrapper initialization.
    L_loderunner = Lightning_LodeRunner(
        lode_runner,
        in_vars=x_vars,
        out_vars=out_vars,
        lr_scheduler=CosineWithWarmupScheduler,
        scheduler_params={
            "warmup_steps": 500,
            "anchor_lr": 1e-3,
            "terminal_steps": 1000,
            "num_cycles": 0.5,
            "min_fraction": 0.5,
            "last_epoch": 0,
        },
    )
    L_loderunner_out = L_loderunner(x, lead_times)
    print("Lightning LodeRunner-tiny output shape:", L_loderunner_out.shape)

    # Small size
    embed_dim = 96
    block_structure = (1, 1, 9, 1)

    lode_runner = LodeRunner(
        default_vars=default_vars,
        image_size=image_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        emb_factor=emb_factor,
        num_heads=num_heads,
        block_structure=block_structure,
        window_sizes=window_sizes,
        patch_merge_scales=patch_merge_scales,
        verbose=False,
    ).to(device)
    print(
        "LodeRunner-small parameters:", count_torch_params(lode_runner, trainable=True)
    )

    # Big size
    embed_dim = 128
    block_structure = (1, 1, 9, 1)

    lode_runner = LodeRunner(
        default_vars=default_vars,
        image_size=image_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        emb_factor=emb_factor,
        num_heads=num_heads,
        block_structure=block_structure,
        window_sizes=window_sizes,
        patch_merge_scales=patch_merge_scales,
        verbose=False,
    ).to(device)
    print("LodeRunner-big parameters:", count_torch_params(lode_runner, trainable=True))

    # Large size
    embed_dim = 192
    block_structure = (1, 1, 9, 1)

    lode_runner = LodeRunner(
        default_vars=default_vars,
        image_size=image_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        emb_factor=emb_factor,
        num_heads=num_heads,
        block_structure=block_structure,
        window_sizes=window_sizes,
        patch_merge_scales=patch_merge_scales,
        verbose=False,
    ).to(device)
    print(
        "LodeRunner-large parameters:", count_torch_params(lode_runner, trainable=True)
    )

    # Huge size
    embed_dim = 352
    block_structure = (1, 1, 9, 1)

    lode_runner = LodeRunner(
        default_vars=default_vars,
        image_size=image_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        emb_factor=emb_factor,
        num_heads=num_heads,
        block_structure=block_structure,
        window_sizes=window_sizes,
        patch_merge_scales=patch_merge_scales,
        verbose=False,
    ).to(device)
    print("LodeRunner-huge parameters:", count_torch_params(lode_runner, trainable=True))

    # Giant size
    embed_dim = 512
    block_structure = (1, 1, 11, 2)

    lode_runner = LodeRunner(
        default_vars=default_vars,
        image_size=image_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        emb_factor=emb_factor,
        num_heads=num_heads,
        block_structure=block_structure,
        window_sizes=window_sizes,
        patch_merge_scales=patch_merge_scales,
        verbose=False,
    ).to(device)
    print(
        "LodeRunner-giant parameters:", count_torch_params(lode_runner, trainable=True)
    )
