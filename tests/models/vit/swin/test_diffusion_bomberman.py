"""Tests for DiffusionBomberman architecture.

This module contains comprehensive unit tests for the DiffusionLodeRunner
model and its Lightning wrapper, including tests for initialization, forward
passes, training/validation steps, and DDIM sampling.
"""

import pytest
from lightning.pytorch import Trainer
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler

from yoke.models.vit.swin.diffusion_bomberman import (
    DiffusionLodeRunner,
    Lightning_DiffusionLodeRunner,
)
from yoke.utils.diffusion.noise_schedulers import VPCosineNoiseSchedule


# Ignore only SLURM "srun" warning
pytestmark = pytest.mark.filterwarnings(
    "ignore: The `srun` command is available on your system but is not used"
)


class MockScheduler(_LRScheduler):
    """Mock of Scheduler class."""

    def __init__(self, optimizer: torch.optim.Optimizer, **kwargs: dict) -> None:
        """Initialization."""
        super().__init__(optimizer)

    def get_lr(self) -> list[float]:
        """Essential method of _LRScheduler."""
        return [group["lr"] for group in self.optimizer.param_groups]


@pytest.fixture
def diffusion_loderunner_model() -> DiffusionLodeRunner:
    """Create DiffusionLodeRunner model fixture."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return DiffusionLodeRunner(
        default_vars=["var1", "var2", "var3"],
        image_size=(1120, 800),
        patch_size=(10, 10),
        embed_dim=96,
        emb_factor=2,
        num_heads=8,
        block_structure=(1, 1, 3, 1),
        window_sizes=[(8, 8), (8, 8), (4, 4), (2, 2)],
        patch_merge_scales=[(2, 2), (2, 2), (2, 2)],
        verbose=False,
    ).to(device)


@pytest.fixture
def lightning_diffusion_model(
    diffusion_loderunner_model: DiffusionLodeRunner,
) -> Lightning_DiffusionLodeRunner:
    """Create Lightning_DiffusionLodeRunner model fixture."""
    lightning_model = Lightning_DiffusionLodeRunner(
        model=diffusion_loderunner_model,
        in_vars=torch.tensor([0, 1, 2]),
        out_vars=torch.tensor([0, 1, 2]),
        lr_scheduler=MockScheduler,
        scheduler_params={"dummy_param": 1},
        loss_fn=nn.MSELoss(),
        noise_schedule=VPCosineNoiseSchedule(),
    )

    lightning_model.trainer = Trainer(logger=False)

    return lightning_model


@pytest.fixture
def noise_schedule() -> VPCosineNoiseSchedule:
    """Create VPCosineNoiseSchedule fixture."""
    return VPCosineNoiseSchedule()


def test_diffusion_loderunner_init(
    diffusion_loderunner_model: DiffusionLodeRunner,
) -> None:
    """Test DiffusionLodeRunner initialization."""
    assert isinstance(diffusion_loderunner_model, DiffusionLodeRunner)
    assert diffusion_loderunner_model.embed_dim == 96
    assert len(diffusion_loderunner_model.default_vars) == 3
    assert diffusion_loderunner_model.image_size == (1120, 800)
    assert diffusion_loderunner_model.patch_size == (10, 10)


def test_diffusion_loderunner_forward(
    diffusion_loderunner_model: DiffusionLodeRunner,
) -> None:
    """Test DiffusionLodeRunner forward pass."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create test inputs
    batch_size = 2
    x = torch.randn(batch_size, 3, 1120, 800).to(device)
    y_tau = torch.randn(batch_size, 2, 1120, 800).to(device)
    in_vars = torch.tensor([0, 1, 2]).to(device)
    out_vars = torch.tensor([0, 1]).to(device)
    lead_times = torch.rand(batch_size).to(device)
    diffusion_time = torch.rand(batch_size).to(device)

    # Forward pass
    output = diffusion_loderunner_model(
        x=x,
        y_tau=y_tau,
        in_vars=in_vars,
        out_vars=out_vars,
        lead_times=lead_times,
        diffusion_time=diffusion_time,
    )

    # Assertions
    assert isinstance(output, torch.Tensor)
    assert output.shape[0] == batch_size
    assert output.shape[1] == len(out_vars)
    assert output.shape[2:] == (1120, 800)
    assert not torch.isnan(output).any()


def test_lightning_diffusion_model_init(
    lightning_diffusion_model: Lightning_DiffusionLodeRunner,
) -> None:
    """Test Lightning_DiffusionLodeRunner initialization."""
    assert isinstance(lightning_diffusion_model, Lightning_DiffusionLodeRunner)
    assert isinstance(lightning_diffusion_model.model, DiffusionLodeRunner)
    assert isinstance(lightning_diffusion_model.noise_schedule, VPCosineNoiseSchedule)


def test_lightning_diffusion_model_configure_optimizers(
    lightning_diffusion_model: Lightning_DiffusionLodeRunner,
) -> None:
    """Test optimizer configuration."""
    optimizer_config = lightning_diffusion_model.configure_optimizers()

    assert "optimizer" in optimizer_config
    assert "lr_scheduler" in optimizer_config
    assert isinstance(optimizer_config["optimizer"], torch.optim.AdamW)


def test_training_step(
    lightning_diffusion_model: Lightning_DiffusionLodeRunner,
) -> None:
    """Test Lightning training step."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create test batch (simulating dataset output)
    batch_size = 2
    x = torch.randn(batch_size, 3, 1120, 800).to(device)
    y = torch.randn(batch_size, 3, 1120, 800).to(device)
    lead_times = torch.rand(batch_size).to(device)

    # Simulate what the dataset does: sample tau and apply forward diffusion
    tau = torch.rand(batch_size).to(device)
    y_tau, noise = lightning_diffusion_model.noise_schedule.forward_diffusion(y, tau)

    # Create batch as dataset would return it
    batch = (x, y_tau, noise, lead_times, tau)

    # Execute training step
    loss = lightning_diffusion_model.training_step(batch, batch_idx=0)

    # Assertions
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Scalar loss
    assert not torch.isnan(loss)
    assert loss.item() >= 0  # MSE loss should be non-negative


def test_validation_step(
    lightning_diffusion_model: Lightning_DiffusionLodeRunner,
) -> None:
    """Test Lightning validation step."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create test batch (simulating dataset output)
    batch_size = 2
    x = torch.randn(batch_size, 3, 1120, 800).to(device)
    y = torch.randn(batch_size, 3, 1120, 800).to(device)
    lead_times = torch.rand(batch_size).to(device)

    # Simulate what the dataset does: sample tau and apply forward diffusion
    tau = torch.rand(batch_size).to(device)
    y_tau, noise = lightning_diffusion_model.noise_schedule.forward_diffusion(y, tau)

    # Create batch as dataset would return it
    batch = (x, y_tau, noise, lead_times, tau)

    # Execute validation step (returns None)
    result = lightning_diffusion_model.validation_step(batch, batch_idx=0)

    assert result is None


def test_ddim_sampling(
    lightning_diffusion_model: Lightning_DiffusionLodeRunner,
) -> None:
    """Test DDIM sampling method."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create test inputs
    batch_size = 2
    x = torch.randn(batch_size, 3, 1120, 800).to(device)
    lead_times = torch.rand(batch_size).to(device)

    # Sample with few steps for speed
    samples = lightning_diffusion_model.sample(
        x=x,
        lead_times=lead_times,
        num_steps=5,
        eta=0.0,
    )

    # Assertions
    assert isinstance(samples, torch.Tensor)
    assert samples.shape[0] == batch_size
    assert samples.shape[1] == len(lightning_diffusion_model.out_vars)
    assert samples.shape[2:] == (1120, 800)
    assert not torch.isnan(samples).any()
