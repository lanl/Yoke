"""Tests for LodeRunner epoch training utilities."""
# ruff: noqa: E402
# Due to pytest warnings needing to be filtered prior to other imports.

import pytest

pytestmark = pytest.mark.filterwarnings(
    "ignore:Detected call of.*lr_scheduler.step.*:UserWarning"
)

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pytest import MonkeyPatch

import yoke.utils.training.epoch.loderunner as epoch_mod


class DummyEpochStep:
    """Helper callable that counts invocations and returns fixed loss tensors."""

    def __init__(self) -> None:
        """Initialize the DummyEpochStep with a call counter."""
        self.calls = 0

    def __call__(
        self, *args: object, **kwargs: object
    ) -> tuple[None, None, torch.Tensor]:
        """Simulate a datastep: returns (truth, pred, loss_tensor)."""
        self.calls += 1
        # return dummy truth/pred (ignored) and a single-sample loss
        return None, None, torch.tensor([0.5], dtype=torch.float32)


@pytest.fixture
def simple_loaders() -> tuple[DataLoader, DataLoader]:
    """Create two DataLoaders each yielding two dummy 3-tuples of Tensors."""
    # Each sample is (start_img, end_img, Dt)
    # shapes here are arbitrary – our DummyEpochStep ignores them
    sample = (
        torch.zeros((1, 1, 1, 1), dtype=torch.float32),
        torch.zeros((1, 1, 1, 1), dtype=torch.float32),
        torch.zeros((1, 1), dtype=torch.float32),
    )
    # two such samples -> two batches at batch_size=1
    data = [sample, sample]
    loader = DataLoader(data, batch_size=1)
    return loader, loader


@pytest.fixture
def dummy_model_optimizer() -> tuple[nn.Module, torch.optim.Optimizer]:
    """Return a dummy model object and a real optimizer with one parameter."""
    model = nn.Linear(1, 1)  # simple linear model
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    return model, optimizer


@pytest.fixture
def loss_fn() -> object:
    """Return a dummy loss function (not actually called)."""
    return lambda *args, **kwargs: None


def test_train_simple_loderunner_epoch(
    tmp_path: Path,
    simple_loaders: tuple[DataLoader, DataLoader],
    dummy_model_optimizer: tuple[object, torch.optim.Optimizer],
    loss_fn: object,
    monkeypatch: MonkeyPatch,
) -> None:
    """Test that train_simple writes both train and val CSV files."""
    train_loader, val_loader = simple_loaders
    model, optimizer = dummy_model_optimizer

    # Patch the imported datastep functions
    fake_train = DummyEpochStep()
    fake_eval = DummyEpochStep()
    monkeypatch.setattr(epoch_mod, "train_loderunner_datastep", fake_train)
    monkeypatch.setattr(epoch_mod, "eval_loderunner_datastep", fake_eval)

    # Filenames with placeholder
    train_file = str(tmp_path / "train_<epochIDX>.csv")
    val_file = str(tmp_path / "val_<epochIDX>.csv")

    # Run with epochIDX=1 and train_per_val=1 => both train & val run
    epoch_mod.train_simple_loderunner_epoch(
        channel_map=[0],
        training_data=train_loader,
        validation_data=val_loader,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochIDX=1,
        train_per_val=1,
        train_rcrd_filename=train_file,
        val_rcrd_filename=val_file,
        device=torch.device("cpu"),
        verbose=False,
    )

    # Check that files exist and have two lines each
    tpath = tmp_path / "train_0001.csv"
    vpath = tmp_path / "val_0001.csv"
    assert tpath.exists(), "Train CSV not created"
    assert vpath.exists(), "Val CSV not created"

    assert len(tpath.read_text().splitlines()) == 2
    assert len(vpath.read_text().splitlines()) == 2

    # Ensure the fake steps were invoked once per batch
    assert fake_train.calls == 2
    assert fake_eval.calls == 2


def test_train_scheduled_loderunner_epoch(
    tmp_path: Path,
    simple_loaders: tuple[DataLoader, DataLoader],
    dummy_model_optimizer: tuple[object, torch.optim.Optimizer],
    loss_fn: object,
    monkeypatch: MonkeyPatch,
) -> None:
    """Test scheduled epoch writes files and returns same scheduled_prob."""
    train_loader, val_loader = simple_loaders
    model, optimizer = dummy_model_optimizer

    fake_train = DummyEpochStep()
    fake_eval = DummyEpochStep()
    monkeypatch.setattr(epoch_mod, "train_scheduled_loderunner_datastep", fake_train)
    monkeypatch.setattr(epoch_mod, "eval_scheduled_loderunner_datastep", fake_eval)

    # Filenames
    tf = str(tmp_path / "train_<epochIDX>.csv")
    vf = str(tmp_path / "val_<epochIDX>.csv")

    sched = 0.7
    out = epoch_mod.train_scheduled_loderunner_epoch(
        training_data=train_loader,
        validation_data=val_loader,
        model=model,
        optimizer=optimizer,
        LRsched=optim.lr_scheduler.StepLR(optimizer, step_size=1),
        loss_fn=loss_fn,
        epochIDX=2,
        train_per_val=2,
        train_rcrd_filename=tf,
        val_rcrd_filename=vf,
        device=torch.device("cpu"),
        scheduled_prob=sched,
    )
    # Returns identified scheduled_prob
    assert out == sched

    # Since 2 % 2 == 0, validation runs
    assert (tmp_path / "train_0002.csv").exists()
    assert (tmp_path / "val_0002.csv").exists()

    assert fake_train.calls == 2
    assert fake_eval.calls == 2


def test_train_LRsched_loderunner_epoch(
    tmp_path: Path,
    simple_loaders: tuple[DataLoader, DataLoader],
    dummy_model_optimizer: tuple[object, torch.optim.Optimizer],
    loss_fn: object,
    monkeypatch: MonkeyPatch,
) -> None:
    """Test LR-scheduler epoch writes CSVs and steps scheduler."""
    train_loader, val_loader = simple_loaders
    model, optimizer = dummy_model_optimizer

    fake_train = DummyEpochStep()
    fake_eval = DummyEpochStep()
    monkeypatch.setattr(epoch_mod, "train_loderunner_datastep", fake_train)
    monkeypatch.setattr(epoch_mod, "eval_loderunner_datastep", fake_eval)

    tf = str(tmp_path / "train_<epochIDX>.csv")
    vf = str(tmp_path / "val_<epochIDX>.csv")

    epoch_mod.train_LRsched_loderunner_epoch(
        channel_map=[0],
        training_data=train_loader,
        validation_data=val_loader,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochIDX=3,
        LRsched=optim.lr_scheduler.StepLR(optimizer, step_size=1),
        train_per_val=1,
        train_rcrd_filename=tf,
        val_rcrd_filename=vf,
        device=torch.device("cpu"),
        verbose=False,
    )
    assert (tmp_path / "train_0003.csv").exists()
    assert (tmp_path / "val_0003.csv").exists()
    assert fake_train.calls == 2
    assert fake_eval.calls == 2


def test_train_DDP_loderunner_epoch(
    tmp_path: Path,
    simple_loaders: tuple[DataLoader, DataLoader],
    dummy_model_optimizer: tuple[object, torch.optim.Optimizer],
    loss_fn: object,
    monkeypatch: MonkeyPatch,
) -> None:
    """Test DDP epoch writes only for rank 0 and obeys batch limits."""
    train_loader, val_loader = simple_loaders
    model, optimizer = dummy_model_optimizer

    fake_ddp_train = DummyEpochStep()
    fake_ddp_eval = DummyEpochStep()
    monkeypatch.setattr(epoch_mod, "train_DDP_loderunner_datastep", fake_ddp_train)
    monkeypatch.setattr(epoch_mod, "eval_DDP_loderunner_datastep", fake_ddp_eval)

    tf = str(tmp_path / "train_<epochIDX>.csv")
    vf = str(tmp_path / "val_<epochIDX>.csv")

    # Rank 0: should write files, stop after 1 batch
    epoch_mod.train_DDP_loderunner_epoch(
        training_data=train_loader,
        validation_data=val_loader,
        num_train_batches=1,
        num_val_batches=1,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        LRsched=optim.lr_scheduler.StepLR(optimizer, step_size=1),
        epochIDX=4,
        train_per_val=1,
        train_rcrd_filename=tf,
        val_rcrd_filename=vf,
        device=torch.device("cpu"),
        rank=0,
        world_size=2,
    )
    assert (tmp_path / "train_0004.csv").exists()
    assert (tmp_path / "val_0004.csv").exists()
    assert fake_ddp_train.calls == 1
    assert fake_ddp_eval.calls == 1

    # Clean up and rerun as rank=1: no files should be created
    os.remove(tmp_path / "train_0004.csv")
    os.remove(tmp_path / "val_0004.csv")
    fake_ddp_train.calls = 0
    fake_ddp_eval.calls = 0

    epoch_mod.train_DDP_loderunner_epoch(
        training_data=train_loader,
        validation_data=val_loader,
        num_train_batches=1,
        num_val_batches=1,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        LRsched=optim.lr_scheduler.StepLR(optimizer, step_size=1),
        epochIDX=5,
        train_per_val=2,
        train_rcrd_filename=tf,
        val_rcrd_filename=vf,
        device=torch.device("cpu"),
        rank=1,
        world_size=2,
    )
    assert not (tmp_path / "train_0005.csv").exists()
    assert not (tmp_path / "val_0005.csv").exists()
    assert fake_ddp_train.calls == 1
    assert fake_ddp_eval.calls == 0
