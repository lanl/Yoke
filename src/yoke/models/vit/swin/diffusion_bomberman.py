"""Module for DiffusionLodeRunner - Score-based Diffusion extension of LodeRunner.

Implements a conditional score-based diffusion model that represents a full
conditional distribution over future fields rather than a single point estimate.
Follows the variance-preserving (VP) forward diffusion process with dual-stream
tokenization for conditioning and noised target.

"""

from collections.abc import Callable, Iterable

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
    DiffusionTimeEmbed,
)
from yoke.utils.diffusion.noise_schedulers import VPCosineNoiseSchedule
from yoke.lr_schedulers import CosineWithWarmupScheduler
from yoke.helpers.training_design import validate_patch_and_window


class DiffusionLodeRunner(nn.Module):
    """DiffusionLodeRunner neural network.

    Score-based diffusion model extending LodeRunner with dual-stream tokenization.
    Implements variance-preserving (VP) forward diffusion and learns to predict
    noise for denoising score matching.

    The model conditions on:
    - Input fields x (conditioning stream)
    - Noised target fields y_tau (noised-target stream)
    - Lead time delta_t (Δt) (temporal offset)
    - Diffusion time tau (τ) ∈ [0,1] (noise level)

    Args:
        default_vars (list[str]): List of default variables to be used for training.
        image_size (tuple[int, int]): Height and width, in pixels, of input image.
        patch_size (tuple[int, int]): Height and width pixel dimensions of patch in
                                      initial embedding.
        embed_dim (int): Initial embedding dimension.
        emb_factor (int): Scale of embedding in each patch merge/expand.
        num_heads (int): Number of heads in the MSA layers.
        block_structure (tuple[int, int, int, int]): Tuple specifying the number of SWIN
                                                     encoders in each block structure
                                                     separated by the patch-merge layers.
        window_sizes (list[tuple[int, int]]): Window sizes within each
                                                SWIN encoder/decoder.
        patch_merge_scales (list[tuple[int, int]]): Height and width scales used in
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
        verbose: bool = False,
    ) -> None:
        """Initialization for DiffusionLodeRunner."""
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

        # ===== Stream A: Conditioning stream (input fields x) =====
        # Parallel patch embedding for conditioning variables
        self.parallel_embed_x = ParallelVarPatchEmbed(
            max_vars=self.max_vars,
            img_size=self.image_size,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            norm_layer=None,
        )

        # Variable embedding for conditioning stream
        self.var_embed_x = VarEmbed(self.default_vars, self.embed_dim)

        # Variable aggregation for conditioning stream
        self.agg_vars_x = AggVars(self.embed_dim, self.num_heads)

        # Position embedding for conditioning stream
        self.pos_embed_x = PosEmbed(
            self.embed_dim,
            self.patch_size,
            self.image_size,
            self.parallel_embed_x.num_patches,
        )

        # ===== Stream B: Noised-target stream (noised fields y_tau) =====
        # Parallel patch embedding for noised target variables
        self.parallel_embed_y_tau = ParallelVarPatchEmbed(
            max_vars=self.max_vars,
            img_size=self.image_size,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            norm_layer=None,
        )

        # Variable embedding for noised-target stream
        self.var_embed_y_tau = VarEmbed(self.default_vars, self.embed_dim)

        # Variable aggregation for noised-target stream
        self.agg_vars_y_tau = AggVars(self.embed_dim, self.num_heads)

        # Position embedding for noised-target stream
        self.pos_embed_y_tau = PosEmbed(
            self.embed_dim,
            self.patch_size,
            self.image_size,
            self.parallel_embed_y_tau.num_patches,
        )

        # ===== Temporal conditioning =====
        # Lead-time encoding (Δt)
        self.temporal_encoding = TimeEmbed(self.embed_dim)

        # Diffusion-time encoding (τ)
        self.diffusion_time_encoding = DiffusionTimeEmbed(self.embed_dim)

        # ===== SWIN U-Net backbone =====
        self.unet = SwinUnetBackbone(
            emb_size=self.embed_dim,
            emb_factor=self.emb_factor,
            patch_grid_size=self.parallel_embed_x.grid_size,
            block_structure=self.block_structure,
            num_heads=self.num_heads,
            window_sizes=self.window_sizes,
            patch_merge_scales=self.patch_merge_scales,
            verbose=verbose,
        )

        # ===== Decoding to noise prediction =====
        # Linear embed the last dimension into V*p_h*p_w for noise prediction
        self.linear4unpatch = nn.Linear(
            self.embed_dim, self.max_vars * self.patch_size[0] * self.patch_size[1]
        )

        # Unmap the tokenized embeddings to variables and images
        self.unpatch = Unpatchify(
            total_num_vars=self.max_vars,
            patch_grid_size=self.parallel_embed_x.grid_size,
            patch_size=self.patch_size,
        )

    def forward(
        self,
        x: torch.Tensor,
        y_tau: torch.Tensor,
        in_vars: torch.Tensor,
        out_vars: torch.Tensor,
        lead_times: torch.Tensor,
        diffusion_time: torch.Tensor,
    ) -> torch.Tensor:
        """Forward method for DiffusionLodeRunner.

        Args:
            x: Conditioning input fields of shape (B, C_in, H, W).
            y_tau: Noised target fields of shape (B, C_out, H, W).
            in_vars: Tensor of variable indices for input (conditioning) variables.
            out_vars: Tensor of variable indices for output (target) variables.
            lead_times: Lead time values of shape (B,) for temporal conditioning.
            diffusion_time: Diffusion time values of shape (B,) in [0, 1].

        Returns:
            Predicted noise tensor of shape (B, C_out, H, W).
        """
        # ===== Stream A: Process conditioning input x =====
        # Embed conditioning input
        z_x = self.parallel_embed_x(x, in_vars)  # (B, N, D)

        # Encode conditioning variables
        z_x = self.var_embed_x(z_x, in_vars)  # (B, N, D)

        # Aggregate conditioning variables
        z_x = self.agg_vars_x(z_x)  # (B, N, D)

        # Encode patch positions for conditioning
        z_x = self.pos_embed_x(z_x)  # (B, N, D)

        # ===== Stream B: Process noised target y_tau =====
        # Embed noised target
        z_y_tau = self.parallel_embed_y_tau(y_tau, out_vars)  # (B, N, D)

        # Encode target variables
        z_y_tau = self.var_embed_y_tau(z_y_tau, out_vars)  # (B, N, D)

        # Aggregate target variables
        z_y_tau = self.agg_vars_y_tau(z_y_tau)  # (B, N, D)

        # Encode patch positions for target
        z_y_tau = self.pos_embed_y_tau(z_y_tau)  # (B, N, D)

        # ===== Token fusion: Additive combination of streams =====
        z = z_x + z_y_tau  # (B, N, D)

        # ===== Temporal conditioning =====
        # Encode lead time Δt
        z = self.temporal_encoding(z, lead_times)  # (B, N, D)

        # Encode diffusion time τ
        z = self.diffusion_time_encoding(z, diffusion_time)  # (B, N, D)

        # ===== SWIN U-Net backbone =====
        z = self.unet(z)  # (B, N, D)

        # ===== Decode to noise prediction =====
        # Linear map to per-variable patch pixels
        z = self.linear4unpatch(z)  # (B, N, V*P_h*P_w)

        # Unpatchify to full resolution
        epsilon_pred = self.unpatch(z)  # (B, V, H, W)

        # Select only output variables (noise prediction for target variables)
        epsilon_pred = epsilon_pred[:, out_vars]  # (B, C_out, H, W)

        return epsilon_pred


class Lightning_DiffusionLodeRunner(LightningModule):
    """Lightning wrapper for DiffusionLodeRunner.

    Wraps DiffusionLodeRunner in a LightningModule for training with
    denoising score matching objective.

    Args:
        model (nn.Module): Pre-initialized DiffusionLodeRunner model.
        in_vars (torch.Tensor): Input variable indices for conditioning.
        out_vars (torch.Tensor): Output variable indices for prediction.
        lr_scheduler (_LRScheduler): Learning-rate scheduler class.
        scheduler_params (dict): Keyword arguments for scheduler initialization.
        loss_fn (Callable): Loss function for noise prediction (default: MSE).
        noise_schedule (VPCosineNoiseSchedule): VP noise schedule for diffusion.
    """

    def __init__(
        self,
        model: nn.Module,
        in_vars: torch.Tensor = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
        out_vars: torch.Tensor = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
        lr_scheduler: _LRScheduler = None,
        scheduler_params: dict = None,
        loss_fn: Callable = nn.MSELoss(),
        noise_schedule: VPCosineNoiseSchedule = None,
    ) -> None:
        """Initialize Lightning wrapper."""
        super().__init__()
        self.model = model
        self.lr_scheduler = lr_scheduler or CosineWithWarmupScheduler
        self.scheduler_params = scheduler_params or {}
        self.loss_fn = loss_fn
        self.noise_schedule = noise_schedule or VPCosineNoiseSchedule()

        # Register buffers for device management
        self.register_buffer("in_vars", in_vars)
        self.register_buffer("out_vars", out_vars)

    def configure_optimizers(self) -> dict:
        """Setup optimizer with scheduler."""
        optimizer = torch.optim.AdamW(self.model.parameters())
        scheduler = self.lr_scheduler(optimizer, **self.scheduler_params)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Execute training step with denoising score matching.

        Args:
            batch: Tuple of (x, y_tau, noise, lead_times, tau) where:
                x: Conditioning input of shape (B, C_in, H, W).
                y_tau: Noised target of shape (B, C_out, H, W).
                noise: Ground truth noise of shape (B, C_out, H, W).
                lead_times: Lead time values of shape (B,).
                tau: Diffusion time values of shape (B,) in [0, 1].
            batch_idx: Batch index.

        Returns:
            Loss value.
        """
        x, y_tau, noise, lead_times, tau = batch

        # Predict noise
        noise_pred = self.model(
            x=x,
            y_tau=y_tau,
            in_vars=self.in_vars,
            out_vars=self.out_vars,
            lead_times=lead_times,
            diffusion_time=tau,
        )

        # Compute loss (MSE between predicted and true noise)
        loss = self.loss_fn(noise_pred, noise)
        batch_loss = loss.mean()

        # Log metrics
        if hasattr(self, "trainer") and self.trainer.training:
            self.log("train_loss", batch_loss, sync_dist=True)

        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        """Execute validation step.

        Args:
            batch: Tuple of (x, y_tau, noise, lead_times, tau) where:
                x: Conditioning input of shape (B, C_in, H, W).
                y_tau: Noised target of shape (B, C_out, H, W).
                noise: Ground truth noise of shape (B, C_out, H, W).
                lead_times: Lead time values of shape (B,).
                tau: Diffusion time values of shape (B,) in [0, 1].
            batch_idx: Batch index.
        """
        x, y_tau, noise, lead_times, tau = batch

        # Predict noise
        noise_pred = self.model(
            x=x,
            y_tau=y_tau,
            in_vars=self.in_vars,
            out_vars=self.out_vars,
            lead_times=lead_times,
            diffusion_time=tau,
        )

        # Compute loss
        loss = self.loss_fn(noise_pred, noise)
        batch_loss = loss.mean()

        # Log metrics
        if hasattr(self, "trainer") and self.trainer.training:
            self.log("train_loss", batch_loss, sync_dist=True)

    @torch.no_grad()
    def sample(
        self,
        x: torch.Tensor,
        lead_times: torch.Tensor,
        num_steps: int = 50,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """Sample from the learned conditional distribution using DDIM.

        Args:
            x: Conditioning input of shape (B, C_in, H, W).
            lead_times: Lead time values of shape (B,).
            num_steps: Number of denoising steps.
            eta: DDIM stochasticity parameter (0 = deterministic).

        Returns:
            Sampled predictions of shape (B, C_out, H, W).
        """
        batch_size = x.shape[0]
        device = x.device

        # Initialize from pure noise
        # Determine output shape from out_vars
        num_out_vars = len(self.out_vars)
        y_tau = torch.randn(
            batch_size, num_out_vars, *self.model.image_size, device=device
        )

        # Create reverse diffusion schedule
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

        for i in range(num_steps):
            tau_current = timesteps[i]
            tau_next = timesteps[i + 1]

            # Broadcast to batch
            tau_batch = tau_current.repeat(batch_size)

            # Predict noise
            noise_pred = self.model(
                x=x,
                y_tau=y_tau,
                in_vars=self.in_vars,
                out_vars=self.out_vars,
                lead_times=lead_times,
                diffusion_time=tau_batch,
            )

            # Predict x0
            y0_pred = self.noise_schedule.remove_noise(y_tau, tau_batch, noise_pred)

            # DDIM update (deterministic when eta=0)
            if tau_next > 0:
                tau_next_batch = tau_next.repeat(batch_size)
                alpha_next = self.noise_schedule.alpha(tau_next_batch.view(-1, 1, 1, 1))
                sigma_next = self.noise_schedule.sigma(tau_next_batch.view(-1, 1, 1, 1))

                # DDIM formula: y_{t-1} = α_{t-1}*ŷ_0 + σ_{t-1}*ε̂
                y_tau = alpha_next * y0_pred + sigma_next * noise_pred
            else:
                # Final step: return predicted x0
                y_tau = y0_pred

            # NOTE: the case where eta>0 (DDPM) is not implemented here.

        return y_tau


if __name__ == "__main__":
    from yoke.utils.parameters import count_torch_params

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Torch Device:", device)

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

    # Target for diffusion (same shape as x for this test)
    y = torch.rand(5, 4, 1120, 800)
    y = y.type(torch.FloatTensor).to(device)

    lead_times = torch.rand(5).to(device)  # Lead time for each entry in batch
    diffusion_times = torch.rand(5).to(device)  # Diffusion time tau in [0, 1]

    # Variable indices
    x_vars = torch.tensor([1, 7, 10, 13]).to(device)
    out_vars = torch.tensor([1, 7, 10, 13]).to(device)

    # Common model setup for DiffusionLodeRunner
    emb_factor = 2
    patch_size = (10, 10)
    image_size = (1120, 800)
    num_heads = 8
    window_sizes = [(8, 8), (8, 8), (4, 4), (2, 2)]
    patch_merge_scales = [(2, 2), (2, 2), (2, 2)]

    # Tiny size
    embed_dim = 96
    block_structure = (1, 1, 3, 1)

    # Test DiffusionLodeRunner architecture
    print("\n" + "=" * 60)
    print("Testing DiffusionLodeRunner Architecture")
    print("=" * 60)

    diffusion_lode_runner = DiffusionLodeRunner(
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

    # Test forward pass (noise prediction)
    noise_pred = diffusion_lode_runner(
        x=x,
        y_tau=y,
        in_vars=x_vars,
        out_vars=out_vars,
        lead_times=lead_times,
        diffusion_time=diffusion_times,
    )
    print(f"\nDiffusionLodeRunner-tiny output shape: {noise_pred.shape}")
    print(f"DiffusionLodeRunner-tiny output has NaNs: {torch.isnan(noise_pred).any()}")
    print(
        f"DiffusionLodeRunner-tiny parameters: "
        f"{count_torch_params(diffusion_lode_runner, trainable=True):,}"
    )

    # Test Lightning wrapper initialization
    print("\n" + "-" * 60)
    print("Testing Lightning Wrapper")
    print("-" * 60)

    L_diffusion_loderunner = Lightning_DiffusionLodeRunner(
        diffusion_lode_runner,
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

    # Test training step (manually simulate dataset output)
    # Sample diffusion times
    batch_size = x.shape[0]
    tau = torch.rand(batch_size, device=x.device)

    # Apply forward diffusion (simulating what dataset does)
    y_tau, noise = L_diffusion_loderunner.noise_schedule.forward_diffusion(y, tau)

    # Create batch as dataset would return it
    batch = (x, y_tau, noise, lead_times, tau)
    x_batch, y_tau_batch, noise_batch, lead_times_batch, tau_batch = batch

    # Predict noise
    noise_pred = L_diffusion_loderunner.model(
        x=x_batch,
        y_tau=y_tau_batch,
        in_vars=L_diffusion_loderunner.in_vars,
        out_vars=L_diffusion_loderunner.out_vars,
        lead_times=lead_times_batch,
        diffusion_time=tau_batch,
    )

    # Compute loss
    loss = L_diffusion_loderunner.loss_fn(noise_pred, noise_batch)
    print(f"\nTraining step loss: {loss.item():.6f}")

    # Test sampling
    print("\n" + "-" * 60)
    print("Testing DDIM Sampling")
    print("-" * 60)

    samples = L_diffusion_loderunner.sample(
        x=x,
        lead_times=lead_times,
        num_steps=10,  # Use fewer steps for testing
        eta=0.0,
    )
    print(f"\nSampled output shape: {samples.shape}")
    print(f"Sampled output has NaNs: {torch.isnan(samples).any()}")

    # Test different model sizes
    print("\n" + "=" * 60)
    print("Testing Different Model Sizes")
    print("=" * 60)

    sizes = [
        ("small", 96, (1, 1, 9, 1)),
        ("big", 128, (1, 1, 9, 1)),
        ("large", 192, (1, 1, 9, 1)),
        ("huge", 352, (1, 1, 9, 1)),
        ("giant", 512, (1, 1, 11, 2)),
    ]

    for size_name, embed_dim, block_structure in sizes:
        diffusion_lode_runner = DiffusionLodeRunner(
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
        param_count = count_torch_params(diffusion_lode_runner, trainable=True)
        print(f"\nDiffusionLodeRunner-{size_name} parameters: {param_count:,}")

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)
