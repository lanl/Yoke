"""Tests for Bomberman architecture."""

import pytest

from lightning.pytorch import Trainer
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler

from yoke.models.vit.swin.bomberman import LodeRunner, Lightning_LodeRunner


class MockScheduler(_LRScheduler):
    """Mock of Scheduler class."""

    def __init__(self, optimizer: torch.optim.Optimizer, **kwargs: dict) -> None:
        """Initialization."""
        super().__init__(optimizer)

    def get_lr(self) -> list[float]:
        """Essential method of _LRScheduler."""
        return [group["lr"] for group in self.optimizer.param_groups]


@pytest.fixture
def loderunner_model() -> LodeRunner:
    """Fixture for LodeRunner tests."""
    return LodeRunner(
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
    )


@pytest.fixture
def lightning_model(loderunner_model: LodeRunner) -> Lightning_LodeRunner:
    """Fixture for Lightning_LodeRunner tests."""
    lightning_loderunner = Lightning_LodeRunner(
        model=loderunner_model,
        in_vars=torch.tensor([0, 1, 2]),
        out_vars=torch.tensor([0, 1, 2]),
        lr_scheduler=MockScheduler,
        scheduler_params={"dummy_param": 1},
        loss_fn=nn.MSELoss(reduction="none"),
        scheduled_sampling_scheduler=lambda global_step: 1.0,
    )
    lightning_loderunner.trainer = Trainer(logger=False)
    return lightning_loderunner


def test_loderunner_init(loderunner_model: LodeRunner) -> None:
    """Test initialization."""
    assert isinstance(loderunner_model, LodeRunner)
    assert loderunner_model.embed_dim == 96
    assert len(loderunner_model.default_vars) == 3


def test_loderunner_forward(loderunner_model: LodeRunner) -> None:
    """Test forward method."""
    x = torch.randn(2, 3, 1120, 800)  # Batch size of 2, 3 channels, image size
    in_vars = torch.tensor([0, 1, 2])
    out_vars = torch.tensor([0, 1])
    lead_times = torch.rand(2)

    output = loderunner_model(x, in_vars, out_vars, lead_times)
    assert isinstance(output, torch.Tensor)
    assert output.shape[0] == 2  # Batch size
    assert output.shape[1] == len(out_vars)  # Number of output variables


def test_lightning_model_init(lightning_model: Lightning_LodeRunner) -> None:
    """Test initialization."""
    assert isinstance(lightning_model, Lightning_LodeRunner)
    assert isinstance(lightning_model.model, LodeRunner)


def test_lightning_model_forward(lightning_model: Lightning_LodeRunner) -> None:
    """Test forward."""
    x = torch.randn(2, 3, 1120, 800)  # Batch size of 2, 3 channels, image size
    lead_times = torch.rand(2)

    output = lightning_model(x, lead_times)
    assert isinstance(output, torch.Tensor)


def test_training_step(lightning_model: Lightning_LodeRunner) -> None:
    """Test lightning training step."""
    batch = (
        torch.randn(2, 2, 3, 1120, 800),  # img_seq
        torch.rand(2),  # lead_times
    )

    batch_loss = lightning_model.training_step(batch, batch_idx=0)
    assert isinstance(batch_loss, torch.Tensor)


def test_validation_step(lightning_model: Lightning_LodeRunner) -> None:
    """Test lightning validation step."""
    batch = (
        torch.randn(2, 2, 3, 1120, 800),  # img_seq
        torch.rand(2),  # lead_times
    )

    lightning_model.validation_step(batch, batch_idx=0)
