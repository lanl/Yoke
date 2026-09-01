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
        predict_delta (bool): When True, the output head predicts a CHANGE relative
            to the last observed magnitude per band (anchored regression) instead of
            an absolute magnitude. See ``__init__`` for the anchoring rule. Adds no
            parameters, so it is backward-compatible with existing checkpoints.
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
        predict_delta: bool = False,
        trend_decay_anchor: bool = False,
        trend_slope_k: int = 3,
        trend_max_offset: float = None,
        pool_mode: str = "mean",
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
                ``2 * dt_fourier_bands + 1``-wide encoding
                ``[sin(Dt·f), cos(Dt·f), log1p(Dt)]`` -- a fixed log-spaced
                frequency bank (periods ~0.5 -> 60 days) plus one non-periodic
                monotone channel -- is concatenated onto BOTH MLP inputs, so the
                trainable path can learn a smooth, nonlinear, per-band dependence
                on lead time (e.g. late-time decay) rather than a
                lead-time-independent persistence value. The frequency bank is a
                non-trainable buffer. The monotone ``log1p(Dt)`` channel gives an
                explicit long-term trend so the forecast does not curve back up
                at long lead times (which a purely periodic basis would).
            predict_delta (bool): When True (default False), the output head predicts
                a normalized-space DELTA that is added to a per-band anchor -- the
                most-recent observed value in that band within the context window --
                so the forecast starts AT the last observation at lead time 0 instead
                of reconstructing the absolute magnitude from scratch. Bands with no
                observation in the window fall back to the most-recent observation in
                ANY band (global-last); if the whole window is empty the anchor is 0
                (the normalized mean). The anchor is derived from ``x`` inside
                ``forward`` and requires the ``valid`` column, so this mode is only
                valid in time-window mode (``context_window_days`` set). It adds NO
                parameters, so the ``state_dict`` is byte-identical to the absolute
                model and existing checkpoints load unchanged; the flag is recorded
                in the checkpoint and restored by the loaders.
            trend_decay_anchor (bool): When True (default False), the per-band delta
                anchor is EXTRAPOLATED along that band's recent local slope instead of
                held flat: ``anchor[b] = v_last[b] + slope[b] * Dt``. The slope is a
                least-squares fit of value vs. ``rel_t`` over that band's most-recent
                ``trend_slope_k`` valid events (bands with < 2 valid events get slope 0,
                i.e. the flat-hold behavior). This lets the forecast lean into a fade
                rather than plateau at the last value. Requires ``predict_delta`` (it
                augments the same anchor) and window mode. The slope is RAW (unclamped):
                for bands whose recent points are pre-peak and rising this extrapolates
                continued brightening, so the output head must learn a residual to
                correct it. Adds NO parameters (derived from ``x``/``Dt``), so the
                ``state_dict`` is unchanged and existing checkpoints load
                ``strict=True``.
            trend_slope_k (int): Number of most-recent valid events per band used to fit
                the local slope when ``trend_decay_anchor`` is on. Default 3.
            trend_max_offset (float): When set (and ``trend_decay_anchor`` is on), the
                extrapolated anchor offset ``slope[b] * Dt`` is symmetrically clamped to
                ``[-trend_max_offset, +trend_max_offset]`` (in per-band normalized /
                z-score units) before being added to ``v_last``. Values are z-scored per
                band, so this bounds the anchor displacement to a fixed number of
                standard deviations regardless of fade direction or lead time -- capping
                the overshoot that a steep raw slope produces at large ``Dt`` (the head
                would otherwise have to learn a large corrective residual, flooring the
                loss). It is deliberately SIGN-AGNOSTIC: it limits both an over-fast fade
                and the near-peak pre-peak "brightening" extrapolation of the raw slope.
                When None (default) the offset is RAW (unclamped), matching the original
                trend-anchor behavior. Adds NO parameters; round-trips via checkpoint.
            pool_mode (str): How the backbone output image [B, C, H, W] is collapsed
                to the per-channel summary the output head consumes. ``"mean"``
                (default) is a global spatial average -- byte-identical to the legacy
                model, so old checkpoints load strict=True. ``"meanstdmax"``
                concatenates the spatial mean, std, and max per channel, tripling the
                head's real input width (``3 * backbone_channels`` instead of
                ``backbone_channels``) so it surfaces spatial structure the mean alone
                discards. This CHANGES the output_head's first-layer shape, so a
                checkpoint trained with a different ``pool_mode`` will NOT load that
                layer (the rest of the head/conditioner/backbone still load); the flag
                is recorded in the checkpoint and restored by the loaders.
        """
        super().__init__()

        if predict_delta and context_window_days is None:
            raise ValueError(
                "predict_delta=True requires time-window mode "
                "(context_window_days set); the per-band anchor needs the "
                "'valid' column present only in the window layout."
            )

        if trend_decay_anchor and not predict_delta:
            raise ValueError(
                "trend_decay_anchor=True requires predict_delta=True; the trend "
                "slope augments the per-band delta anchor."
            )

        self.backbone = backbone
        self.context_len = context_len
        self.n_bands = n_bands
        self.image_size = image_size
        self.backbone_channels = backbone_channels
        self.context_window_days = context_window_days
        self.dt_fourier_bands = dt_fourier_bands
        self.predict_delta = predict_delta
        self.trend_decay_anchor = trend_decay_anchor
        self.trend_slope_k = trend_slope_k
        self.trend_max_offset = trend_max_offset

        if pool_mode not in ("mean", "meanstdmax"):
            raise ValueError(
                f"pool_mode must be 'mean' or 'meanstdmax', got {pool_mode!r}."
            )
        self.pool_mode = pool_mode
        # Number of pooled statistics per backbone channel fed to the output head.
        pool_channels = backbone_channels * (3 if pool_mode == "meanstdmax" else 1)

        # Dataset x layout, flattened per event. Fixed-count mode:
        #   [value, rel_t, one_hot_band(n_bands)] * context_len   -> 2 + n_bands
        # Time-window mode adds a validity flag so padding is carried in x:
        #   [value, rel_t, valid, one_hot_band(n_bands)] * context_len -> 3 + n_bands
        per_event_width = 3 + n_bands if context_window_days is not None else 2 + n_bands
        input_dim = context_len * per_event_width

        # Fourier lead-time encoding. When enabled, a fixed log-spaced frequency
        # bank turns the scalar Dt into a 2*dt_fourier_bands feature vector that
        # the trainable MLPs consume directly. Registered as a buffer so it
        # saves/loads and moves with .to(device) but is excluded from the
        # optimizer's param list.
        #
        # The longest period is 60 days, well beyond any plausible forecast
        # horizon: a purely sinusoidal basis MUST turn around (each component
        # bottoms at half its period), which produced an unphysical "smile" in
        # the late-time forecast (fade, then rise) as the lowest-frequency band
        # -- previously 15 d, ~one full cycle over the window -- swung back up.
        # With a 60 d max period the slowest component covers only a fraction of
        # a cycle across a <=15 d horizon, so it stays monotone. In addition, a
        # single non-periodic log1p(Dt) channel is appended (see _encode_dt),
        # giving the MLPs an explicit monotone ramp for the long-term trend while
        # the Fourier bank handles short-timescale structure.
        if dt_fourier_bands > 0:
            periods = torch.logspace(
                math.log10(0.5), math.log10(60.0), dt_fourier_bands
            )
            self.register_buffer("dt_freqs", 2.0 * math.pi / periods)
            # 2 * bands (sin + cos) + 1 monotone log1p(Dt) channel.
            dt_extra = 2 * dt_fourier_bands + 1
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
            nn.Linear(pool_channels + dt_extra, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_bands),
        )

    def _encode_dt(self, Dt: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Fourier-encode the lead time for the trainable path.

        Args:
            Dt (torch.Tensor): Lead-time tensor of shape [B] (or broadcastable).
            batch_size (int): Batch size B, used to size the disabled-path output.

        Returns:
            torch.Tensor: [B, 2 * dt_fourier_bands + 1] of
            ``[sin(Dt·f), cos(Dt·f), log1p(Dt)]`` when enabled, else an empty
            [B, 0] tensor (so the concat is a no-op and the disabled path is
            identical to the legacy model). The trailing ``log1p(Dt)`` is a
            non-periodic, monotone channel so the trainable path always has an
            explicit "further ahead -> keep fading" ramp, independent of the
            periodic bands (which alone would eventually curve back up).
        """
        if self.dt_fourier_bands == 0:
            return Dt.new_zeros((batch_size, 0))

        Dt = Dt.reshape(batch_size, 1)
        # [B, 1] * [1, bands] -> [B, bands]
        angles = Dt * self.dt_freqs.reshape(1, -1)
        mono = torch.log1p(Dt.clamp_min(0.0))  # [B, 1], monotone in lead time
        return torch.cat([torch.sin(angles), torch.cos(angles), mono], dim=1)

    def _band_anchor(self, x: torch.Tensor, Dt: torch.Tensor) -> torch.Tensor:
        """Per-band anchor from the windowed context, for delta mode.

        Reconstructs, for each band, the most-recent observed (normalized) value in
        the trailing context window. The window-mode ``x`` flattens per event as
        ``[value, rel_t, valid, one_hot_band(n_bands)]`` (see the dataset's
        ``_getitem_window``); "most recent" is the valid event of that band with the
        largest ``rel_t``. Bands with no observation in the window fall back to the
        most-recent observation in ANY band (global-last). A fully empty window (no
        valid events) yields an all-zero anchor (the normalized mean).

        When ``self.trend_decay_anchor`` is set, the per-band anchor is additionally
        extrapolated along that band's recent local slope,
        ``anchor[b] = v_last[b] + slope[b] * Dt``, where ``slope[b]`` is a
        least-squares fit of value vs. ``rel_t`` over that band's most-recent
        ``self.trend_slope_k`` valid events. Bands with < 2 valid events get slope 0
        (flat hold). When ``self.trend_max_offset`` is set, the offset ``slope[b] * Dt``
        is symmetrically clamped to ``[-trend_max_offset, +trend_max_offset]`` (z-score
        units) so a steep slope cannot overshoot at large ``Dt``; when None the slope is
        RAW (unclamped) and a pre-peak/rising band extrapolates continued brightening.
        ``Dt`` and ``rel_t`` share the same time units (days), so the extrapolation is
        unit-consistent.

        Args:
            x (torch.Tensor): Window-mode context, shape
                [B, context_len * (3 + n_bands)].
            Dt (torch.Tensor): Lead time (days) from the anchor (most-recent event) to
                the target, shape [B] or broadcastable. Only used when
                ``self.trend_decay_anchor``; the flat-hold path ignores it.

        Returns:
            torch.Tensor: Per-band anchor of shape [B, n_bands] in normalized units.
        """
        B = x.shape[0]
        nb = self.n_bands
        ev = x.view(B, self.context_len, 3 + nb)  # [B, L, 3+nb]

        value = ev[..., 0]  # [B, L]
        rel_t = ev[..., 1]  # [B, L]
        valid = ev[..., 2] > 0.5  # [B, L] bool
        band_oh = ev[..., 3:]  # [B, L, nb]
        band_idx = band_oh.argmax(dim=-1)  # [B, L]

        neg_inf = torch.finfo(rel_t.dtype).min

        # Global-last: value of the valid event with the largest rel_t (any band).
        g_score = torch.where(valid, rel_t, torch.full_like(rel_t, neg_inf))
        g_any = valid.any(dim=1)  # [B]
        g_arg = g_score.argmax(dim=1)  # [B]
        global_last = value.gather(1, g_arg.unsqueeze(1)).squeeze(1)  # [B]
        global_last = torch.where(g_any, global_last, torch.zeros_like(global_last))

        # Per-band last: for each band b, the valid event of band b with max rel_t.
        # match[b] over events, scored by rel_t; [B, nb, L].
        band_match = (
            valid.unsqueeze(1)
            & (band_idx.unsqueeze(1) == torch.arange(nb, device=x.device).view(1, nb, 1))
        )  # [B, nb, L]
        pb_score = torch.where(
            band_match,
            rel_t.unsqueeze(1),
            torch.full_like(rel_t.unsqueeze(1), neg_inf),
        )  # [B, nb, L]
        pb_has = band_match.any(dim=2)  # [B, nb]
        pb_arg = pb_score.argmax(dim=2)  # [B, nb]
        per_band_last = torch.gather(
            value.unsqueeze(1).expand(B, nb, self.context_len), 2, pb_arg.unsqueeze(2)
        ).squeeze(2)  # [B, nb]

        # Fall back to global-last where a band was never observed in the window.
        v_last = torch.where(pb_has, per_band_last, global_last.unsqueeze(1))

        if not self.trend_decay_anchor:
            return v_last

        # Trend/decay anchor: extrapolate each band along the local slope of its
        # most-recent trend_slope_k valid events, anchor = v_last + slope * Dt.
        # Select those events per band as the top-k by rel_t (pb_score already masks
        # non-matching/invalid events to -inf), then least-squares-fit value vs rel_t.
        k = min(self.trend_slope_k, self.context_len)
        topk_t, topk_idx = pb_score.topk(k, dim=2)  # [B, nb, k]; -inf where < k matches
        # Robust validity of each selected slot: gather the band-match boolean at the
        # chosen indices (top-k of an all -inf row lands on non-matching events).
        topk_valid = torch.gather(
            band_match.to(value.dtype), 2, topk_idx
        ) > 0.5  # [B, nb, k]
        topk_v = torch.gather(
            value.unsqueeze(1).expand(B, nb, self.context_len), 2, topk_idx
        )  # [B, nb, k]

        # Weighted (valid-only) least-squares slope of v vs t per (batch, band).
        w = topk_valid.to(value.dtype)  # [B, nb, k]
        t = torch.where(topk_valid, topk_t, torch.zeros_like(topk_t))
        v = torch.where(topk_valid, topk_v, torch.zeros_like(topk_v))
        n = w.sum(dim=2)  # [B, nb]
        denom = n.clamp_min(1.0)
        t_bar = (w * t).sum(dim=2) / denom  # [B, nb]
        v_bar = (w * v).sum(dim=2) / denom  # [B, nb]
        dt_ = t - t_bar.unsqueeze(2)
        dv_ = v - v_bar.unsqueeze(2)
        cov = (w * dt_ * dv_).sum(dim=2)  # [B, nb]
        var = (w * dt_ * dt_).sum(dim=2)  # [B, nb]
        slope = torch.where(
            (n >= 2.0) & (var > 1e-8), cov / var.clamp_min(1e-8), torch.zeros_like(cov)
        )  # [B, nb]; bands with < 2 valid events -> 0 (flat hold)

        # Extrapolated offset in z-score units. Symmetrically clamp its MAGNITUDE when
        # trend_max_offset is set: a steep raw slope times a large Dt would otherwise
        # push the anchor far past the data, forcing the head to learn a large
        # corrective residual (which floors the loss). The cap is sign-agnostic, so it
        # bounds both an over-fast fade and the near-peak pre-peak brightening the raw
        # slope produces. When None, the offset is raw (unclamped).
        offset = slope * Dt.reshape(B, 1)
        if self.trend_max_offset is not None:
            offset = offset.clamp(-self.trend_max_offset, self.trend_max_offset)
        return v_last + offset

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

        # Collapse spatial dimensions to backbone-channel summaries. The mean
        # alone throws away all spatial structure the backbone produced; the
        # meanstdmax mode also carries the per-channel spatial spread and peak,
        # tripling the head's real input width for near-zero cost.
        if self.pool_mode == "meanstdmax":
            pred_channel_vals = torch.cat(
                [
                    pred_img.mean(dim=(2, 3)),
                    pred_img.flatten(2).std(dim=2),
                    pred_img.amax(dim=(2, 3)),
                ],
                dim=1,
            )  # [B, 3 * backbone_channels]
        else:
            pred_channel_vals = pred_img.mean(dim=(2, 3))  # [B, backbone_channels]

        # Convert backbone channels to per-band predictions, conditioning the
        # head on lead time via the same Fourier encoding ([B, 0] when disabled).
        pred = self.output_head(
            torch.cat([pred_channel_vals, dt_feat], dim=1)
        )  # [B, n_bands]

        # Delta mode: the head predicts a change relative to the per-band last
        # observed value (anchored regression), so the forecast starts at the last
        # observation at Dt=0 instead of reconstructing the absolute magnitude. The
        # anchor is derived from x (no parameters), so the disabled path is
        # byte-identical to the absolute model.
        if self.predict_delta:
            pred = pred + self._band_anchor(x, Dt)

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
