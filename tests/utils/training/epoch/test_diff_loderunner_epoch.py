"""Tests for the DiffusionLodeRunner epoch training/evaluation utilities."""
# ruff: noqa: E402
# Filter scheduler warnings before importing torch scheduler machinery.

import pytest

pytestmark = pytest.mark.filterwarnings(
    "ignore:Detected call of.*lr_scheduler.step.*:UserWarning"
)

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import yoke.utils.training.epoch.diff_loderunner as epoch_mod


class DummyDatastep:
    """Callable stand-in for a datastep returning fixed per-sample losses."""

    def __init__(self, batch_size: int = 1) -> None:
        """Record call count and the per-sample loss length to return.

        Args:
            batch_size (int): Number of per-sample loss values to return.
        """
        self.calls = 0
        self.batch_size = batch_size

    def __call__(
        self, *args: object, **kwargs: object
    ) -> tuple[None, None, torch.Tensor]:
        """Return ``(noise_gt, noise_pred, per_sample_loss)`` with dummy values."""
        self.calls += 1
        return None, None, torch.full((self.batch_size,), 0.5, dtype=torch.float32)


class DummyLRSched:
    """Learning-rate scheduler stub that counts ``step`` calls."""

    def __init__(self) -> None:
        """Initialize the step counter."""
        self.steps = 0

    def step(self) -> None:
        """Increment the recorded step count."""
        self.steps += 1


@pytest.fixture
def loaders() -> tuple[DataLoader, DataLoader]:
    """Return two DataLoaders each yielding two batches at batch_size=1."""
    sample = (
        torch.zeros((1, 8, 2, 2)),
        torch.zeros((1, 8, 2, 2)),
        torch.ones((1, 8, 2, 2)),
        torch.ones((1,)),
        torch.full((1,), 0.5),
    )
    data = [sample, sample]
    loader = DataLoader(data, batch_size=1)
    return loader, loader


@pytest.fixture
def model_optimizer() -> tuple[nn.Module, torch.optim.Optimizer]:
    """Return a trivial model and a real optimizer."""
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return model, optimizer


def _count_rows(path: Path) -> int:
    """Return the number of data rows in a saved record CSV.

    Args:
        path (Path): Path to the CSV record file.

    Returns:
        int: Number of rows.
    """
    arr = np.loadtxt(path, delimiter=",")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return len(arr)


def test_train_simple_epoch_with_validation(
    tmp_path: Path,
    loaders: tuple[DataLoader, DataLoader],
    model_optimizer: tuple[nn.Module, torch.optim.Optimizer],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """train_simple writes both train and val CSVs when epochIDX % train_per_val == 0."""
    train_loader, val_loader = loaders
    model, optimizer = model_optimizer

    fake_train = DummyDatastep(batch_size=1)
    fake_eval = DummyDatastep(batch_size=1)
    monkeypatch.setattr(epoch_mod, "train_diffusion_loderunner_datastep", fake_train)
    monkeypatch.setattr(epoch_mod, "eval_diffusion_loderunner_datastep", fake_eval)

    train_file = str(tmp_path / "train_<epochIDX>.csv")
    val_file = str(tmp_path / "val_<epochIDX>.csv")

    epoch_mod.train_simple_diffusion_loderunner_epoch(
        training_data=train_loader,
        validation_data=val_loader,
        model=model,
        optimizer=optimizer,
        loss_fn=nn.MSELoss(reduction="none"),
        epochIDX=1,
        train_per_val=1,
        train_rcrd_filename=train_file,
        val_rcrd_filename=val_file,
        device=torch.device("cpu"),
        in_vars=torch.arange(8),
        out_vars=torch.arange(8),
        verbose=True,
    )

    train_out = tmp_path / "train_0001.csv"
    val_out = tmp_path / "val_0001.csv"
    assert train_out.exists() and val_out.exists()
    assert fake_train.calls == 2 and fake_eval.calls == 2
    assert _count_rows(train_out) == 2
    assert _count_rows(val_out) == 2


def test_train_simple_epoch_skips_validation(
    tmp_path: Path,
    loaders: tuple[DataLoader, DataLoader],
    model_optimizer: tuple[nn.Module, torch.optim.Optimizer],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """train_simple skips validation when epochIDX % train_per_val != 0."""
    train_loader, val_loader = loaders
    model, optimizer = model_optimizer

    fake_train = DummyDatastep(batch_size=1)
    fake_eval = DummyDatastep(batch_size=1)
    monkeypatch.setattr(epoch_mod, "train_diffusion_loderunner_datastep", fake_train)
    monkeypatch.setattr(epoch_mod, "eval_diffusion_loderunner_datastep", fake_eval)

    train_file = str(tmp_path / "train_<epochIDX>.csv")
    val_file = str(tmp_path / "val_<epochIDX>.csv")

    epoch_mod.train_simple_diffusion_loderunner_epoch(
        training_data=train_loader,
        validation_data=val_loader,
        model=model,
        optimizer=optimizer,
        loss_fn=nn.MSELoss(reduction="none"),
        epochIDX=1,
        train_per_val=2,  # 1 % 2 != 0 -> no validation
        train_rcrd_filename=train_file,
        val_rcrd_filename=val_file,
        device=torch.device("cpu"),
        in_vars=torch.arange(8),
        out_vars=torch.arange(8),
        verbose=False,
    )

    assert (tmp_path / "train_0001.csv").exists()
    assert not (tmp_path / "val_0001.csv").exists()
    assert fake_eval.calls == 0


@pytest.mark.parametrize("rank", [0, 1])
def test_train_DDP_epoch(
    rank: int,
    tmp_path: Path,
    loaders: tuple[DataLoader, DataLoader],
    model_optimizer: tuple[nn.Module, torch.optim.Optimizer],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DDP epoch steps the scheduler, honors num_batches, writes CSVs on rank 0."""
    train_loader, val_loader = loaders
    model, optimizer = model_optimizer

    fake_train = DummyDatastep(batch_size=2)
    fake_eval = DummyDatastep(batch_size=2)
    monkeypatch.setattr(epoch_mod, "train_DDP_diffusion_loderunner_datastep", fake_train)
    monkeypatch.setattr(epoch_mod, "eval_DDP_diffusion_loderunner_datastep", fake_eval)

    lrsched = DummyLRSched()
    train_file = str(tmp_path / "ddp_train_<epochIDX>.csv")
    val_file = str(tmp_path / "ddp_val_<epochIDX>.csv")

    epoch_mod.train_DDP_diffusion_loderunner_epoch(
        training_data=train_loader,
        validation_data=val_loader,
        num_train_batches=1,  # break after a single batch
        num_val_batches=1,
        model=model,
        optimizer=optimizer,
        loss_fn=nn.MSELoss(reduction="none"),
        LRsched=lrsched,
        epochIDX=2,
        train_per_val=1,
        train_rcrd_filename=train_file,
        val_rcrd_filename=val_file,
        device=torch.device("cpu"),
        rank=rank,
        world_size=1,
        in_vars=torch.arange(8),
        out_vars=torch.arange(8),
    )

    # Only one train and one val batch processed due to num_*_batches limits.
    assert fake_train.calls == 1
    assert fake_eval.calls == 1
    assert lrsched.steps == 1

    train_out = tmp_path / "ddp_train_0002.csv"
    val_out = tmp_path / "ddp_val_0002.csv"
    if rank == 0:
        assert train_out.exists() and val_out.exists()
        assert _count_rows(train_out) == 2  # batch_size=2 loss values
        assert _count_rows(val_out) == 2
    else:
        assert not train_out.exists()
        assert not val_out.exists()


def test_eval_diffusion_epoch(
    tmp_path: Path,
    loaders: tuple[DataLoader, DataLoader],
    model_optimizer: tuple[nn.Module, torch.optim.Optimizer],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Eval-only epoch writes a test CSV and honors num_test_batches."""
    test_loader, _ = loaders
    model, _ = model_optimizer

    fake_eval = DummyDatastep(batch_size=1)
    monkeypatch.setattr(epoch_mod, "eval_diffusion_loderunner_datastep", fake_eval)

    test_file = str(tmp_path / "test_<epochIDX>.csv")

    epoch_mod.eval_diffusion_loderunner_epoch(
        testing_data=test_loader,
        num_test_batches=1,  # break after a single batch
        model=model,
        loss_fn=nn.MSELoss(reduction="none"),
        epochIDX=3,
        test_rcrd_filename=test_file,
        device=torch.device("cpu"),
        in_vars=torch.arange(8),
        out_vars=torch.arange(8),
    )

    test_out = tmp_path / "test_0003.csv"
    assert test_out.exists()
    assert fake_eval.calls == 1
    assert _count_rows(test_out) == 1
