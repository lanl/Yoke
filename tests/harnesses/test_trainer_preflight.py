"""Tests for HarnessTrainer.preflight and the loss-device move.

Unlike ``test_trainer.py``, the preflight tests use a real one-process ``gloo``
group, real DDP wrapping, real distributed dataloaders, and real Yoke epoch
functions, because checking that wiring is the point of preflight.
"""

import argparse
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import Dataset

import yoke.harnesses.trainer as trainer_mod
from yoke.harnesses.trainer import HarnessTrainer
from yoke.helpers import cli
from yoke.utils.builders import build_from_checkpoint
from yoke.utils.checkpointing import load_model_and_optimizer
from yoke.utils.parallel import setup_single_process_cpu
from yoke.utils.training.epoch.array_output import train_DDP_array_epoch
from yoke.utils.training.epoch.loderunner import train_DDP_loderunner_epoch


# ---------------------------------------------------------------------------
# Tiny datasets and models matching two batch contracts
# ---------------------------------------------------------------------------
class _ArrayData(Dataset):
    """(inpt, truth) pairs for the array contract."""

    def __len__(self) -> int:
        """Return a small fixed length."""
        return 8

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a 3-vector input and a 2-vector target."""
        return torch.ones(3), torch.zeros(2)


class _LodeRunnerData(Dataset):
    """(start_img, end_img, Dt) triples for the loderunner 1-step contract."""

    def __len__(self) -> int:
        """Return a small fixed length."""
        return 8

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        """Return 2-channel 4x3 images and a scalar lead time."""
        return torch.ones(2, 4, 3), torch.zeros(2, 4, 3), torch.tensor(0.25)


class _TinyLinear(nn.Module):
    """Linear map from 3 to 2 features."""

    def __init__(self) -> None:
        """Create the linear layer."""
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear layer."""
        return self.linear(x)


class _TinyLodeRunner(nn.Module):
    """Adds a learnable scalar; matches the LodeRunner forward signature."""

    def __init__(self) -> None:
        """Create the scalar parameter."""
        super().__init__()
        self.param = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        x: torch.Tensor,
        in_vars: torch.Tensor,
        out_vars: torch.Tensor,
        lead_times: torch.Tensor,
    ) -> torch.Tensor:
        """Return the selected input channels plus the parameter."""
        return x[:, out_vars] + self.param


class _BufferLoss(nn.Module):
    """Elementwise squared error weighted by a registered buffer."""

    def __init__(self) -> None:
        """Register a weight buffer."""
        super().__init__()
        self.register_buffer("weight", torch.tensor(2.0))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return the weighted elementwise squared error."""
        return self.weight * (pred - target) ** 2


def _builder(model_class: type) -> object:
    """Return a fresh-run model_builder for ``model_class``."""

    def build(
        args: argparse.Namespace, device: torch.device
    ) -> tuple[nn.Module, dict, type, int, None]:
        return model_class().to(device), {}, model_class, 0, None

    return build


def _parse(extra: list[str] | None = None) -> argparse.Namespace:
    """Parse args with the real Yoke training builders plus ``--preflight``."""
    parser = argparse.ArgumentParser()
    parser = cli.add_default_args(parser=parser)
    parser = cli.add_training_args(parser=parser)
    return parser.parse_args(["--preflight", "--batch_size", "8", *(extra or [])])


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
def test_preflight_flag_defaults_off() -> None:
    """add_training_args declares --preflight, off by default."""
    parser = cli.add_training_args(argparse.ArgumentParser())
    assert parser.parse_args([]).preflight is False
    assert parser.parse_args(["--preflight"]).preflight is True


def test_preflight_array_contract(tmp_path: Path) -> None:
    """Preflight runs one eval step through train_DDP_array_epoch on CPU."""
    args = _parse(["--trn_rcrd_filename", str(tmp_path / "trn.csv")])
    trainer = HarnessTrainer(
        args,
        model_builder=_builder(_TinyLinear),
        dataset_builder=lambda a: (_ArrayData(), _ArrayData()),
        epoch_fn=train_DDP_array_epoch,
    )
    report = trainer.preflight()

    assert report["model_class"] == "_TinyLinear"
    assert report["parameters"] == 3 * 2 + 2
    # Batch size is capped at 2 for the check.
    assert report["batch"] == ["(2, 3) torch.float32", "(2, 2) torch.float32"]
    assert len(report["losses"]) == 2
    # The process group is torn down afterwards.
    assert not dist.is_initialized()
    # Records went to a temporary directory, not the configured filename.
    assert not (tmp_path / "trn.csv").exists()


def test_preflight_does_not_step_optimizer() -> None:
    """No backward pass or optimizer step happens in preflight."""
    args = _parse()
    trainer = HarnessTrainer(
        args,
        model_builder=_builder(_TinyLinear),
        dataset_builder=lambda a: (_ArrayData(), _ArrayData()),
        epoch_fn=train_DDP_array_epoch,
    )
    trainer.preflight()
    module = trainer.model.module
    assert all(p.grad is None for p in module.parameters())
    assert not trainer.optimizer.state


def test_preflight_loderunner_contract_with_epoch_kwargs() -> None:
    """Preflight passes epoch_kwargs (channel_map, dataset tag) through."""
    args = _parse()
    trainer = HarnessTrainer(
        args,
        model_builder=_builder(_TinyLodeRunner),
        dataset_builder=lambda a: (_LodeRunnerData(), _LodeRunnerData()),
        epoch_fn=train_DDP_loderunner_epoch,
        epoch_kwargs={"channel_map": [0, 1], "dataset": "pli"},
    )
    report = trainer.preflight()
    assert report["batch"][0] == "(2, 2, 4, 3) torch.float32"
    # pred = 1 + 1, target = 0, so the mean squared error is 4.
    assert report["losses"] == pytest.approx([4.0, 4.0])


def test_preflight_reports_contract_mismatch() -> None:
    """A dataset/epoch-function mismatch surfaces as an error, not a pass."""
    args = _parse()
    trainer = HarnessTrainer(
        args,
        model_builder=_builder(_TinyLodeRunner),
        dataset_builder=lambda a: (_ArrayData(), _ArrayData()),
        epoch_fn=train_DDP_loderunner_epoch,
        epoch_kwargs={"channel_map": [0, 1], "dataset": "pli"},
    )
    with pytest.raises(ValueError):
        trainer.preflight()
    assert not dist.is_initialized()


def test_preflight_rejects_non_finite_loss() -> None:
    """A NaN loss fails the preflight."""

    def nan_loss() -> nn.Module:
        class _Nan(nn.Module):
            def forward(self, p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
                return torch.full_like(p, float("nan"))

        return _Nan()

    args = _parse()
    trainer = HarnessTrainer(
        args,
        model_builder=_builder(_TinyLinear),
        dataset_builder=lambda a: (_ArrayData(), _ArrayData()),
        epoch_fn=train_DDP_array_epoch,
        loss_builder=nan_loss,
    )
    with pytest.raises(RuntimeError, match="not finite"):
        trainer.preflight()


def test_preflight_rejects_missing_validation_record() -> None:
    """An epoch function that records nothing fails the preflight."""
    args = _parse()
    trainer = HarnessTrainer(
        args,
        model_builder=_builder(_TinyLinear),
        dataset_builder=lambda a: (_ArrayData(), _ArrayData()),
        epoch_fn=lambda **kwargs: None,
    )
    with pytest.raises(RuntimeError, match="no validation loss"):
        trainer.preflight()


def test_run_dispatches_to_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    """run() calls preflight() and skips the training lifecycle."""
    calls: list[str] = []
    monkeypatch.setattr(HarnessTrainer, "preflight", lambda self: calls.append("pf"))
    monkeypatch.setattr(
        trainer_mod, "setup_distributed", lambda: calls.append("ddp") or None
    )
    trainer = HarnessTrainer(
        _parse(),
        model_builder=_builder(_TinyLinear),
        dataset_builder=lambda a: (_ArrayData(), _ArrayData()),
        epoch_fn=train_DDP_array_epoch,
    )
    trainer.run()
    assert calls == ["pf"]


def test_setup_single_process_cpu() -> None:
    """The helper creates a one-rank gloo group on the CPU."""
    try:
        assert setup_single_process_cpu() == (0, 1, 0, torch.device("cpu"))
        assert dist.get_backend() == "gloo"
        assert dist.get_world_size() == 1
    finally:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Loss device move (known issue 7)
# ---------------------------------------------------------------------------
def test_loss_module_is_moved_to_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """setup() moves a loss module, including its buffers, to the device."""
    moved: list[object] = []
    original_to = _BufferLoss.to

    def recording_to(self: nn.Module, *args: object, **kwargs: object) -> nn.Module:
        moved.extend(args)
        return original_to(self, *args, **kwargs)

    monkeypatch.setattr(_BufferLoss, "to", recording_to)
    trainer = HarnessTrainer(
        _parse(),
        model_builder=_builder(_TinyLinear),
        dataset_builder=lambda a: (_ArrayData(), _ArrayData()),
        epoch_fn=train_DDP_array_epoch,
        loss_builder=_BufferLoss,
    )
    trainer.preflight()
    assert moved == [torch.device("cpu")]
    assert trainer.loss_fn.weight.device.type == "cpu"


def test_plain_callable_loss_is_left_alone() -> None:
    """A non-Module loss callable is used as is."""

    def squared(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (pred - target) ** 2

    trainer = HarnessTrainer(
        _parse(),
        model_builder=_builder(_TinyLinear),
        dataset_builder=lambda a: (_ArrayData(), _ArrayData()),
        epoch_fn=train_DDP_array_epoch,
        loss_builder=lambda: squared,
    )
    trainer.preflight()
    assert trainer.loss_fn is squared


# ---------------------------------------------------------------------------
# CPU single-process mode: a real run() with no CUDA calls
# ---------------------------------------------------------------------------
def _no_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any CUDA call in the trainer module fail the test."""

    def _fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("CUDA was called in CPU mode")

    monkeypatch.setattr(trainer_mod.torch.cuda, "synchronize", _fail)
    monkeypatch.setattr(trainer_mod.torch.cuda, "set_device", _fail)
    monkeypatch.delenv("SLURM_NTASKS", raising=False)


def _cpu_args(tmp_path: Path, extra: list[str] | None = None) -> argparse.Namespace:
    """Parse real training args for a short 2-epoch CPU run in tmp_path."""
    parser = argparse.ArgumentParser()
    parser = cli.add_default_args(parser=parser)
    parser = cli.add_training_args(parser=parser)
    return parser.parse_args(
        [
            "--studyIDX", "7",
            "--batch_size", "4",
            "--total_epochs", "2",
            "--cycle_epochs", "2",
            "--train_batches", "2",
            "--val_batches", "1",
            "--TRAIN_PER_VAL", "1",
            "--num_workers", "0",
            "--trn_rcrd_filename", str(tmp_path / "trn_<epochIDX>.csv"),
            "--val_rcrd_filename", str(tmp_path / "val_<epochIDX>.csv"),
            *(extra or []),
        ]
    )  # fmt: skip


def _constant_lr(
    optimizer: torch.optim.Optimizer, args: argparse.Namespace, last_epoch: int
) -> torch.optim.lr_scheduler.LambdaLR:
    """Constant LR; train_DDP_array_epoch steps a scheduler every batch."""
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)


def test_cpu_mode_full_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """device_type='cpu' trains, records, checkpoints, and tears down."""
    _no_cuda(monkeypatch)
    monkeypatch.chdir(tmp_path)
    trainer = HarnessTrainer(
        _cpu_args(tmp_path),
        model_builder=build_from_checkpoint(_TinyLinear, lambda a: {}),
        dataset_builder=lambda a: (_ArrayData(), _ArrayData()),
        epoch_fn=train_DDP_array_epoch,
        scheduler_builder=_constant_lr,
        resubmit=False,
        device_type="cpu",
    )
    trainer.run()

    assert trainer.device.type == "cpu"
    assert not dist.is_initialized()
    assert (tmp_path / "trn_0001.csv").exists()
    assert (tmp_path / "val_0002.csv").exists()
    ckpt = tmp_path / "study007_modelState_epoch0002.pth"
    assert ckpt.exists()

    model, _, epoch = load_model_and_optimizer(
        str(ckpt),
        torch.optim.AdamW,
        {"lr": 1e-4},
        {"_TinyLinear": _TinyLinear},
        device="cpu",
    )
    assert epoch == 2
    assert isinstance(model, _TinyLinear)


def test_cpu_mode_continuation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A CPU-mode continuation resumes from the checkpoint's epoch."""
    _no_cuda(monkeypatch)
    monkeypatch.chdir(tmp_path)
    common = {
        "model_builder": build_from_checkpoint(_TinyLinear, lambda a: {}),
        "dataset_builder": lambda a: (_ArrayData(), _ArrayData()),
        "epoch_fn": train_DDP_array_epoch,
        "scheduler_builder": _constant_lr,
        "resubmit": False,
        "device_type": "cpu",
    }
    HarnessTrainer(_cpu_args(tmp_path), **common).run()

    ckpt = str(tmp_path / "study007_modelState_epoch0002.pth")
    args = _cpu_args(
        tmp_path,
        ["--total_epochs", "3", "--continuation", "--checkpoint", ckpt],
    )
    trainer = HarnessTrainer(args, **common)
    trainer.run()
    assert trainer.epochIDX == 3
    assert (tmp_path / "study007_modelState_epoch0003.pth").exists()
