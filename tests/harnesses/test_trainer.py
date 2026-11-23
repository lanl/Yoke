"""Tests for yoke.harnesses.trainer.HarnessTrainer and TrainerHooks.

The DDP orchestration is exercised on CPU by monkeypatching the distributed
entry points (``setup_distributed``/``cleanup_distributed``), the DDP wrapper,
the distributed dataloader factory, and the collective sync primitives. This
lets the timed epoch loop, checkpoint save, and resubmission logic be tested
without GPUs or a real process group.
"""

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import yoke.harnesses.trainer as trainer_mod
from yoke.harnesses.trainer import HarnessTrainer, TrainerHooks


class _TinyNet(nn.Module):
    """Minimal model used throughout the trainer tests."""

    def __init__(self, width: int = 3) -> None:
        """Initialize a single linear layer of the given width."""
        super().__init__()
        self.linear = nn.Linear(width, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward map."""
        return self.linear(x)


class _IdentityDDP(nn.Module):
    """Stand-in for DistributedDataParallel that exposes ``.module``."""

    def __init__(self, module: nn.Module, **kwargs: object) -> None:
        """Wrap the module, recording it as ``.module`` like real DDP."""
        super().__init__()
        self.module = module

    def forward(self, *args: object, **kwargs: object) -> object:
        """Delegate to the wrapped module."""
        return self.module(*args, **kwargs)


class _FakeSampler:
    """Records the epochs passed to ``set_epoch``."""

    def __init__(self) -> None:
        """Initialize the epoch record."""
        self.epochs: list[int] = []

    def set_epoch(self, epoch: int) -> None:
        """Record the epoch."""
        self.epochs.append(epoch)


class _FakeDataloader:
    """Minimal dataloader exposing a ``sampler`` with ``set_epoch``."""

    def __init__(self) -> None:
        """Attach a fake sampler."""
        self.sampler = _FakeSampler()


def _make_args(**overrides: object) -> argparse.Namespace:
    """Build a Namespace with the training attributes the trainer needs."""
    base = dict(
        studyIDX=3,
        batch_size=4,
        total_epochs=10,
        cycle_epochs=2,
        train_batches=25,
        val_batches=5,
        TRAIN_PER_VAL=1,
        num_workers=0,
        trn_rcrd_filename="./trn.csv",
        val_rcrd_filename="./val.csv",
        continuation=False,
        checkpoint=None,
        init_learnrate=1e-4,
        submissionType="slurm",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.fixture
def patched_ddp(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Patch the DDP/collective entry points in the trainer module.

    Returns a dict of recorders so tests can assert on saves/resubmissions.
    """
    recorder: dict = {"saves": [], "submits": [], "continuations": [], "evaluations": []}

    monkeypatch.setattr(
        trainer_mod,
        "setup_distributed",
        lambda: (0, 1, 0, torch.device("cpu")),
    )
    monkeypatch.setattr(trainer_mod, "cleanup_distributed", lambda: None)
    monkeypatch.setattr(trainer_mod, "DDP", _IdentityDDP)
    monkeypatch.setattr(
        trainer_mod,
        "make_distributed_dataloader",
        lambda *a, **k: _FakeDataloader(),
    )

    # Neutralize collective sync primitives (no real process group).
    monkeypatch.setattr(trainer_mod.dist, "barrier", lambda *a, **k: None)
    monkeypatch.setattr(trainer_mod.torch.cuda, "synchronize", lambda *a, **k: None)

    def _fake_save(
        model: object,
        optimizer: object,
        epoch: int,
        filepath: str,
        model_class: type,
        model_args: dict,
        extra_state: dict | None = None,
    ) -> None:
        recorder["saves"].append(
            SimpleNamespace(
                epoch=epoch,
                filepath=filepath,
                model_class=model_class,
                model_args=model_args,
                extra_state=extra_state,
            )
        )

    monkeypatch.setattr(trainer_mod, "save_model_and_optimizer", _fake_save)

    def _fake_continuation(
        checkpointpath: str,
        studyIDX: int,
        last_epoch: int,
        submission_type: str = "slurm",
    ) -> str:
        recorder["continuations"].append(
            SimpleNamespace(
                checkpointpath=checkpointpath,
                studyIDX=studyIDX,
                last_epoch=last_epoch,
                submission_type=submission_type,
            )
        )
        return "study003_restart.slurm"

    monkeypatch.setattr(
        trainer_mod.HarnessStudy, "continuation_setup", staticmethod(_fake_continuation)
    )

    # The trainer constructs a HarnessStudy and calls run_evaluation for the
    # final checkpoint. Stub the constructor (avoid filesystem/template checks)
    # and record run_evaluation invocations.
    def _fake_init(
        self: object,
        rundir: str = "./runs",
        template_dir: str = ".",
        cp_file: str = "cp_files.txt",
        submission_type: str = "slurm",
        dryrun: bool = False,
    ) -> None:
        self.submission_type = submission_type.lower()
        self.DRYRUN = dryrun

    monkeypatch.setattr(trainer_mod.HarnessStudy, "__init__", _fake_init)

    def _fake_run_evaluation(
        self: object,
        study_dir: str,
        checkpointpath: str,
        study: dict | None = None,
    ) -> str:
        recorder["evaluations"].append(
            SimpleNamespace(
                study_dir=study_dir,
                checkpointpath=checkpointpath,
                submission_type=self.submission_type,
            )
        )
        return "study003_evaluation.slurm"

    monkeypatch.setattr(trainer_mod.HarnessStudy, "run_evaluation", _fake_run_evaluation)
    monkeypatch.setattr(
        trainer_mod.os, "system", lambda cmd: recorder["submits"].append(cmd)
    )

    return recorder


def _fresh_model_builder(
    args: argparse.Namespace, device: torch.device
) -> tuple[nn.Module, dict, type, int, None]:
    """A model_builder producing a fresh _TinyNet (no optimizer)."""
    model_args = {"width": 3}
    model = _TinyNet(**model_args).to(device)
    return model, model_args, _TinyNet, 0, None


def _dataset_builder(args: argparse.Namespace) -> tuple[object, object]:
    """Return dummy (train, val) datasets; dataloader is faked out anyway."""
    return object(), object()


def test_setup_fresh_builds_optimizer_and_wraps_ddp(patched_ddp: dict) -> None:
    """A fresh run builds an optimizer via optimizer_builder and DDP-wraps."""
    args = _make_args()

    def recording_epoch_fn(**kwargs: object) -> None:
        return None

    trainer = HarnessTrainer(
        args,
        model_builder=_fresh_model_builder,
        dataset_builder=_dataset_builder,
        epoch_fn=recording_epoch_fn,
    )
    trainer.setup_distributed()
    trainer.setup()

    assert isinstance(trainer.model, _IdentityDDP)
    assert isinstance(trainer.optimizer, torch.optim.AdamW)
    assert trainer.model_class is _TinyNet
    assert trainer.starting_epoch == 0
    assert trainer.last_epoch == -1  # fresh run sentinel
    assert isinstance(trainer.loss_fn, nn.MSELoss)


def test_setup_continuation_uses_returned_optimizer(patched_ddp: dict) -> None:
    """On continuation the builder's optimizer is used (state preserved)."""
    args = _make_args(continuation=True, checkpoint="ckpt.pth")
    model = _TinyNet()
    restored_opt = torch.optim.AdamW(model.parameters(), lr=5e-5)

    def continuation_builder(
        a: argparse.Namespace, device: torch.device
    ) -> tuple[nn.Module, dict, type, int, torch.optim.Optimizer]:
        return model.to(device), {"width": 3}, _TinyNet, 4, restored_opt

    trainer = HarnessTrainer(
        args,
        model_builder=continuation_builder,
        dataset_builder=_dataset_builder,
        epoch_fn=lambda **k: None,
    )
    trainer.setup_distributed()
    trainer.setup()

    assert trainer.optimizer is restored_opt
    assert trainer.starting_epoch == 4
    # last_epoch = train_batches * (starting_epoch - 1) = 25 * 3
    assert trainer.last_epoch == 75


def test_train_calls_epoch_fn_expected_times(patched_ddp: dict) -> None:
    """The epoch loop calls epoch_fn once per epoch with the expected kwargs."""
    args = _make_args(total_epochs=10, cycle_epochs=2)
    calls: list[dict] = []

    def recording_epoch_fn(**kwargs: object) -> None:
        calls.append(kwargs)
        return None

    trainer = HarnessTrainer(
        args,
        model_builder=_fresh_model_builder,
        dataset_builder=_dataset_builder,
        epoch_fn=recording_epoch_fn,
        epoch_kwargs={"channel_map": [0, 1, 2], "dataset": "pli"},
    )
    trainer.setup_distributed()
    trainer.setup()
    trainer.train()

    # starting_epoch becomes 1; ending_epoch = min(1+2, 11) = 3 -> epochs 1, 2.
    assert len(calls) == 2
    first = calls[0]
    assert first["epochIDX"] == 1
    assert first["num_train_batches"] == 25
    assert first["num_val_batches"] == 5
    assert first["train_per_val"] == 1
    assert first["channel_map"] == [0, 1, 2]
    assert first["dataset"] == "pli"
    assert first["model"] is trainer.model
    assert first["optimizer"] is trainer.optimizer
    # The sampler should have had set_epoch called for each epoch.
    assert trainer.train_dataloader.sampler.epochs == [1, 2]


def test_global_step_updated_from_epoch_fn_return(patched_ddp: dict) -> None:
    """An int returned by epoch_fn updates the trainer's global_step."""
    args = _make_args(total_epochs=10, cycle_epochs=1)

    def epoch_fn(**kwargs: object) -> int:
        return 4242

    trainer = HarnessTrainer(
        args,
        model_builder=_fresh_model_builder,
        dataset_builder=_dataset_builder,
        epoch_fn=epoch_fn,
    )
    trainer.setup_distributed()
    trainer.setup()
    trainer.train()

    assert trainer.global_step == 4242


def test_finalize_saves_with_matching_model_class(patched_ddp: dict) -> None:
    """Finalize saves .pth with the model_class returned by the builder."""
    args = _make_args(total_epochs=10, cycle_epochs=2)

    trainer = HarnessTrainer(
        args,
        model_builder=_fresh_model_builder,
        dataset_builder=_dataset_builder,
        epoch_fn=lambda **k: None,
    )
    trainer.setup_distributed()
    trainer.setup()
    trainer.train()
    trainer.finalize()

    assert len(patched_ddp["saves"]) == 1
    save = patched_ddp["saves"][0]
    assert save.model_class is _TinyNet
    assert save.model_args == {"width": 3}
    assert save.epoch == 2
    assert save.filepath.endswith("study003_modelState_epoch0002.pth")


def test_finalize_resubmits_when_unfinished(patched_ddp: dict) -> None:
    """Finalize resubmits a continuation job when epochs remain."""
    args = _make_args(total_epochs=10, cycle_epochs=2)

    trainer = HarnessTrainer(
        args,
        model_builder=_fresh_model_builder,
        dataset_builder=_dataset_builder,
        epoch_fn=lambda **k: None,
    )
    trainer.setup_distributed()
    trainer.setup()
    trainer.train()
    trainer.finalize()

    assert len(patched_ddp["continuations"]) == 1
    assert patched_ddp["continuations"][0].last_epoch == 2
    assert patched_ddp["submits"] == ["sbatch study003_restart.slurm"]


def test_finalize_no_resubmit_when_finished(patched_ddp: dict) -> None:
    """No resubmission occurs once the total epoch budget is reached."""
    args = _make_args(total_epochs=2, cycle_epochs=5)

    trainer = HarnessTrainer(
        args,
        model_builder=_fresh_model_builder,
        dataset_builder=_dataset_builder,
        epoch_fn=lambda **k: None,
    )
    trainer.setup_distributed()
    trainer.setup()
    trainer.train()  # runs epochs 1, 2 -> finished
    trainer.finalize()

    assert patched_ddp["continuations"] == []
    assert patched_ddp["submits"] == []


def test_finalize_submits_evaluation_after_final_save(patched_ddp: dict) -> None:
    """A finished opt-in trainer submits one evaluation after its checkpoint."""
    trainer = HarnessTrainer(
        _make_args(total_epochs=2, cycle_epochs=5),
        model_builder=_fresh_model_builder,
        dataset_builder=_dataset_builder,
        epoch_fn=lambda **k: None,
        evaluate_after_training=True,
    )
    trainer.setup_distributed()
    trainer.setup()
    trainer.train()
    trainer.finalize()

    assert len(patched_ddp["saves"]) == 1
    assert len(patched_ddp["evaluations"]) == 1
    evaluation = patched_ddp["evaluations"][0]
    assert evaluation.checkpointpath == "./study003_modelState_epoch0002.pth"
    assert evaluation.study_dir == "."
    assert evaluation.submission_type == "slurm"
    # Submission is routed through run_evaluation, not a bare os.system call
    # in finalize; the continuation os.system path stays unused when finished.
    assert patched_ddp["submits"] == []


def test_finalize_evaluates_ema_companion_when_selected(
    patched_ddp: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """evaluate_checkpoint='ema' targets the _ema.pth companion checkpoint."""
    monkeypatch.chdir(tmp_path)
    # The EMA companion must exist on disk for selection to succeed.
    (tmp_path / "study003_modelState_epoch0002_ema.pth").write_text("ema\n")

    trainer = HarnessTrainer(
        _make_args(total_epochs=2, cycle_epochs=5),
        model_builder=_fresh_model_builder,
        dataset_builder=_dataset_builder,
        epoch_fn=lambda **k: None,
        evaluate_after_training=True,
        evaluate_checkpoint="ema",
    )
    trainer.setup_distributed()
    trainer.setup()
    trainer.train()
    trainer.finalize()

    assert len(patched_ddp["evaluations"]) == 1
    evaluation = patched_ddp["evaluations"][0]
    assert evaluation.checkpointpath == "./study003_modelState_epoch0002_ema.pth"


def test_finalize_ema_selection_requires_companion(
    patched_ddp: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """evaluate_checkpoint='ema' raises when no EMA companion was written."""
    monkeypatch.chdir(tmp_path)  # no _ema.pth companion present

    trainer = HarnessTrainer(
        _make_args(total_epochs=2, cycle_epochs=5),
        model_builder=_fresh_model_builder,
        dataset_builder=_dataset_builder,
        epoch_fn=lambda **k: None,
        evaluate_after_training=True,
        evaluate_checkpoint="ema",
    )
    trainer.setup_distributed()
    trainer.setup()
    trainer.train()
    with pytest.raises(FileNotFoundError, match="EMA companion"):
        trainer.finalize()


def test_trainer_rejects_unknown_evaluate_checkpoint() -> None:
    """The constructor rejects an unsupported evaluate_checkpoint value."""
    with pytest.raises(ValueError, match="Unknown evaluate_checkpoint"):
        HarnessTrainer(
            _make_args(),
            model_builder=_fresh_model_builder,
            dataset_builder=_dataset_builder,
            epoch_fn=lambda **k: None,
            evaluate_checkpoint="swa",
        )


def test_finalize_never_evaluates_unfinished_cycle(patched_ddp: dict) -> None:
    """An unfinished cycle submits only its continuation."""
    trainer = HarnessTrainer(
        _make_args(total_epochs=10, cycle_epochs=2),
        model_builder=_fresh_model_builder,
        dataset_builder=_dataset_builder,
        epoch_fn=lambda **k: None,
        evaluate_after_training=True,
    )
    trainer.setup_distributed()
    trainer.setup()
    trainer.train()
    trainer.finalize()

    assert patched_ddp["evaluations"] == []
    assert len(patched_ddp["continuations"]) == 1


def test_finalize_respects_resubmit_flag(patched_ddp: dict) -> None:
    """resubmit=False disables continuation even when unfinished."""
    args = _make_args(total_epochs=10, cycle_epochs=2)

    trainer = HarnessTrainer(
        args,
        model_builder=_fresh_model_builder,
        dataset_builder=_dataset_builder,
        epoch_fn=lambda **k: None,
        resubmit=False,
    )
    trainer.setup_distributed()
    trainer.setup()
    trainer.train()
    trainer.finalize()

    assert patched_ddp["continuations"] == []
    assert patched_ddp["submits"] == []


def test_hooks_fire_in_expected_order(patched_ddp: dict) -> None:
    """Hooks fire at their defined points and can contribute extra_state."""
    args = _make_args(total_epochs=10, cycle_epochs=2)
    order: list[str] = []

    def on_after_ddp_wrap(t: HarnessTrainer) -> None:
        order.append("ddp_wrap")

    def on_epoch_start(t: HarnessTrainer, epochIDX: int) -> None:
        order.append(f"epoch_start:{epochIDX}")

    def on_after_step(t: HarnessTrainer) -> None:
        order.append("after_step")

    def on_before_save(t: HarnessTrainer, epochIDX: int) -> dict:
        order.append(f"before_save:{epochIDX}")
        return {"global_step": t.global_step}

    def epoch_fn(**kwargs: object) -> int:
        # Simulate the epoch invoking the after-step hook.
        order.append("epoch_fn")
        return 7

    hooks = TrainerHooks(
        on_after_ddp_wrap=on_after_ddp_wrap,
        on_epoch_start=on_epoch_start,
        on_after_step=on_after_step,
        on_before_save=on_before_save,
    )
    trainer = HarnessTrainer(
        args,
        model_builder=_fresh_model_builder,
        dataset_builder=_dataset_builder,
        epoch_fn=epoch_fn,
        hooks=hooks,
    )
    trainer.setup_distributed()
    trainer.setup()
    trainer.train()
    trainer.finalize()

    # DDP wrap hook fires during setup, before any epoch.
    assert order[0] == "ddp_wrap"
    # Each epoch: epoch_start then epoch_fn.
    assert order[1] == "epoch_start:1"
    assert order[2] == "epoch_fn"
    assert order[3] == "epoch_start:2"
    assert order[4] == "epoch_fn"
    # Save hook fires last, during finalize.
    assert order[-1] == "before_save:2"
    # on_before_save contributed extra_state to the checkpoint.
    assert patched_ddp["saves"][0].extra_state == {"global_step": 7}


def test_run_executes_full_lifecycle(patched_ddp: dict) -> None:
    """run() performs setup, train, finalize, and teardown."""
    args = _make_args(total_epochs=10, cycle_epochs=2)
    calls: list[dict] = []

    trainer = HarnessTrainer(
        args,
        model_builder=_fresh_model_builder,
        dataset_builder=_dataset_builder,
        epoch_fn=lambda **k: calls.append(k),
    )
    trainer.run()

    assert len(calls) == 2
    assert len(patched_ddp["saves"]) == 1


def test_setup_before_distributed_raises() -> None:
    """Calling setup() before setup_distributed() is an error."""
    args = _make_args()
    trainer = HarnessTrainer(
        args,
        model_builder=_fresh_model_builder,
        dataset_builder=_dataset_builder,
        epoch_fn=lambda **k: None,
    )
    with pytest.raises(RuntimeError):
        trainer.setup()


def test_scheduler_builder_is_used(patched_ddp: dict) -> None:
    """A provided scheduler_builder is invoked and passed to epoch_fn as LRsched."""
    args = _make_args(total_epochs=10, cycle_epochs=1)
    sentinel_scheduler = object()
    captured: dict = {}

    def scheduler_builder(
        optimizer: object, a: argparse.Namespace, last_epoch: int
    ) -> object:
        captured["last_epoch"] = last_epoch
        return sentinel_scheduler

    calls: list[dict] = []

    trainer = HarnessTrainer(
        args,
        model_builder=_fresh_model_builder,
        dataset_builder=_dataset_builder,
        epoch_fn=lambda **k: calls.append(k),
        scheduler_builder=scheduler_builder,
    )
    trainer.setup_distributed()
    trainer.setup()
    trainer.train()

    assert trainer.scheduler is sentinel_scheduler
    assert captured["last_epoch"] == -1  # fresh run
    assert calls[0]["LRsched"] is sentinel_scheduler


def test_ema_model_and_global_step_threaded_into_epoch(patched_ddp: dict) -> None:
    """When an EMA shadow is set, it and the live global_step reach epoch_fn."""
    args = _make_args(total_epochs=10, cycle_epochs=2)
    sentinel_ema = object()
    calls: list[dict] = []

    def on_after_ddp_wrap(t: HarnessTrainer) -> None:
        # Emulate make_ema_hooks: register a shadow + restore a global_step.
        t.ema_model = sentinel_ema
        t.global_step = 100

    def epoch_fn(**kwargs: object) -> int:
        calls.append(kwargs)
        # Emulate the epoch advancing the global-step counter.
        return kwargs["global_step"] + 5

    trainer = HarnessTrainer(
        args,
        model_builder=_fresh_model_builder,
        dataset_builder=_dataset_builder,
        epoch_fn=epoch_fn,
        hooks=TrainerHooks(on_after_ddp_wrap=on_after_ddp_wrap),
        epoch_kwargs={"channel_map": [0, 1, 2]},
    )
    trainer.setup_distributed()
    trainer.setup()
    trainer.train()

    # Two epochs; ema_model threaded both times.
    assert calls[0]["ema_model"] is sentinel_ema
    assert calls[1]["ema_model"] is sentinel_ema
    # global_step fed in as the live value and advanced across epochs.
    assert calls[0]["global_step"] == 100
    assert calls[1]["global_step"] == 105
    assert trainer.global_step == 110


def test_no_ema_means_no_ema_kwargs(patched_ddp: dict) -> None:
    """Without an EMA shadow, ema_model/global_step are not injected."""
    args = _make_args(total_epochs=10, cycle_epochs=1)
    calls: list[dict] = []

    trainer = HarnessTrainer(
        args,
        model_builder=_fresh_model_builder,
        dataset_builder=_dataset_builder,
        epoch_fn=lambda **k: calls.append(k),
    )
    trainer.setup_distributed()
    trainer.setup()
    trainer.train()

    assert "ema_model" not in calls[0]
    assert "global_step" not in calls[0]


# ---------------------------------------------------------------------------
# device_type (CPU single-process mode)
# ---------------------------------------------------------------------------
def test_invalid_device_type_rejected() -> None:
    """An unknown device_type raises ValueError at construction."""
    with pytest.raises(ValueError, match="device_type"):
        HarnessTrainer(
            _make_args(),
            model_builder=_fresh_model_builder,
            dataset_builder=_dataset_builder,
            epoch_fn=lambda **k: None,
            device_type="tpu",
        )


def test_default_device_type_uses_setup_distributed(patched_ddp: dict) -> None:
    """The default is cuda mode, which calls setup_distributed."""
    trainer = HarnessTrainer(
        _make_args(),
        model_builder=_fresh_model_builder,
        dataset_builder=_dataset_builder,
        epoch_fn=lambda **k: None,
    )
    assert trainer.device_type == "cuda"
    trainer.setup_distributed()  # patched: returns the CPU device
    assert trainer.world_size == 1


def test_cpu_mode_rejects_multiple_slurm_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CPU mode refuses to start when SLURM launched several tasks."""
    monkeypatch.setenv("SLURM_NTASKS", "4")
    called = []
    monkeypatch.setattr(
        trainer_mod, "setup_single_process_cpu", lambda: called.append(1)
    )
    trainer = HarnessTrainer(
        _make_args(),
        model_builder=_fresh_model_builder,
        dataset_builder=_dataset_builder,
        epoch_fn=lambda **k: None,
        device_type="cpu",
    )
    with pytest.raises(RuntimeError, match="--ntasks=1"):
        trainer.setup_distributed()
    assert not called
