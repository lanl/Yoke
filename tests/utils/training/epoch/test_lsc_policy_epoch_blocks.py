"""Extra unit tests for BLOCK_CND paths in train_lsc_policy_epoch.

These tests verify:
  * Gradient monitoring headers and rows are written when ``blocks`` is set
    and ``rank == 0``.
  * Only the gradient header is written when there are zero training batches.
  * No gradient or record files are written when ``rank != 0`` even if
    ``blocks`` is set, while the validation notice still prints.

The tests use a lightweight wrapper that mimics DistributedDataParallel by
providing a ``.module`` attribute. Parameters on ``.module`` are assigned
gradients by dummy train steps to validate RMS computation and recording.
"""

from __future__ import annotations

import math
import pathlib
from collections.abc import Iterable

import pytest
import torch

import yoke.utils.training.epoch.lsc_policy as lsc_policy_module


class _ParamModel(torch.nn.Module):
    """Tiny model with named parameters used for gradient RMS tests.

    This model exposes two parameters whose gradients are controlled by the
    dummy training step to produce predictable RMS values in the gradient
    record file.
    """

    def __init__(self) -> None:
        """Initialize the parameter tensors."""
        super().__init__()
        self.p1 = torch.nn.Parameter(torch.zeros(2), requires_grad=True)
        self.p2 = torch.nn.Parameter(torch.zeros(3), requires_grad=True)


class _DDPishWrapper:
    """Simple wrapper mimicking DistributedDataParallel structure.

    The training function under test accesses parameters via ``model.module``.
    This wrapper provides a ``.module`` attribute pointing at ``_ParamModel``,
    and defines ``train``/``eval`` to match the expected API.
    """

    def __init__(self) -> None:
        """Create the wrapped module instance."""
        self.module = _ParamModel()

    def train(self) -> None:  # noqa: D401 - simple pass-through
        """Set training mode (no-op for this stub)."""
        # No state needed for these tests.

    def eval(self) -> None:  # noqa: D401 - simple pass-through
        """Set eval mode (no-op for this stub)."""
        # No state needed for these tests.


class _DummyScheduler:
    """Minimal learning-rate scheduler stub used to count step calls."""

    def __init__(self) -> None:
        """Initialize the step counter."""
        self.step_count = 0

    def step(self) -> None:
        """Increment the step counter."""
        self.step_count += 1


def _make_blocks() -> list[tuple[str, callable]]:
    """Create a blocks list with one matching and one non-matching selector.

    Returns:
        list[tuple[str, callable]]: A list of block definitions where:
            * "p1" matches parameter name "p1".
            * "nomatch" matches no parameter names.
    """
    p1_block: tuple[str, callable] = ("p1", lambda name: name == "p1")
    nomatch_block: tuple[str, callable] = (
        "nomatch",
        lambda name: str(name).startswith("zzz"),
    )
    return [p1_block, nomatch_block]


def test_blocks_grad_rms_recorded_rank0(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Record RMS gradients and headers when blocks are set and rank is zero.

    This test sets deterministic gradients on two parameters:
      * ``p1.grad = [3, 4]`` which yields an RMS of ``sqrt((3^2+4^2)/2)``.
      * ``p2.grad = [0, 0, 0]`` which does not match any block selector.

    Asserts that the gradient record file contains the expected header and one
    data row with the computed RMS for "p1" and zero for "nomatch".
    """
    device = torch.device("cpu")
    model = _DDPishWrapper()
    optimizer: torch.optim.Optimizer = torch.optim.SGD([torch.zeros(1)], lr=0.1)
    loss_fn: torch.nn.Module = torch.nn.MSELoss()
    scheduler = _DummyScheduler()

    num_train_batches = 1
    num_val_batches = 0
    epoch_idx = 2
    train_per_val = 99
    train_template = str(tmp_path / "train_<epochIDX>.csv")
    val_template = str(tmp_path / "val_<epochIDX>.csv")

    def dummy_train_step(
        _data: Iterable[object],
        mdl: _DDPishWrapper,
        _opt: torch.optim.Optimizer,
        _lfn: torch.nn.Module,
        dev: torch.device,
        _rnk: int,
        _ws: int,
    ) -> tuple[None, None, torch.Tensor]:
        """Set grads and return a single-element loss tensor."""
        mdl.module.p1.grad = torch.tensor([3.0, 4.0], device=dev)
        mdl.module.p2.grad = torch.tensor([0.0, 0.0, 0.0], device=dev)
        return None, None, torch.tensor([0.111], device=dev)

    monkeypatch.setattr(
        lsc_policy_module,
        "train_lsc_policy_datastep",
        dummy_train_step,
    )

    lsc_policy_module.train_lsc_policy_epoch(
        training_data=[1],
        validation_data=[],
        num_train_batches=num_train_batches,
        num_val_batches=num_val_batches,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        LRsched=scheduler,
        epochIDX=epoch_idx,
        train_per_val=train_per_val,
        train_rcrd_filename=train_template,
        val_rcrd_filename=val_template,
        device=device,
        rank=0,
        world_size=1,
        blocks=_make_blocks(),
    )

    assert scheduler.step_count == 1

    grad_path = tmp_path / f"train_{epoch_idx:04d}_grad.csv"
    assert grad_path.exists()
    glines = grad_path.read_text().splitlines()
    assert glines[0] == "epoch,batch,p1,nomatch"

    cols = glines[1].split(",")
    assert cols[0] == str(epoch_idx)
    assert cols[1] == "0"
    rms_p1 = float(cols[2])
    rms_nomatch = float(cols[3])

    expected_p1 = math.sqrt((3.0**2 + 4.0**2) / 2.0)
    # File stores 10 decimal places; compare at that precision.
    assert f"{rms_p1:.10f}" == f"{expected_p1:.10f}"
    assert rms_nomatch == 0.0

    train_path = tmp_path / f"train_{epoch_idx:04d}.csv"
    assert train_path.exists()
    tlines = train_path.read_text().splitlines()
    assert tlines[0] == "epoch,batch,loss"
    assert tlines[1].startswith(f"{epoch_idx}, 0, 0.11100000")


def test_blocks_grad_header_only_when_zero_train_batches(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Write only gradient header and training header for zero train batches.

    Ensures that no data steps are executed when the number of training
    batches is zero. Both the training record file and gradient record file
    should exist and contain only their respective headers.
    """
    device = torch.device("cpu")
    model = _DDPishWrapper()
    optimizer: torch.optim.Optimizer = torch.optim.SGD([torch.zeros(1)], lr=0.1)
    loss_fn: torch.nn.Module = torch.nn.MSELoss()
    scheduler = _DummyScheduler()

    num_train_batches = 0
    num_val_batches = 0
    epoch_idx = 7
    train_per_val = 99
    train_template = str(tmp_path / "train_<epochIDX>.csv")
    val_template = str(tmp_path / "val_<epochIDX>.csv")

    def never_train_step(*args: object, **kwargs: object) -> None:
        """Fail if the train data step is invoked."""
        pytest.fail("train_lsc_policy_datastep should not be called.")

    monkeypatch.setattr(
        lsc_policy_module,
        "train_lsc_policy_datastep",
        never_train_step,
    )

    lsc_policy_module.train_lsc_policy_epoch(
        training_data=[],
        validation_data=[],
        num_train_batches=num_train_batches,
        num_val_batches=num_val_batches,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        LRsched=scheduler,
        epochIDX=epoch_idx,
        train_per_val=train_per_val,
        train_rcrd_filename=train_template,
        val_rcrd_filename=val_template,
        device=device,
        rank=0,
        world_size=1,
        blocks=_make_blocks(),
    )

    grad_path = tmp_path / f"train_{epoch_idx:04d}_grad.csv"
    assert grad_path.exists()
    glines = grad_path.read_text().splitlines()
    assert glines == ["epoch,batch,p1,nomatch"]

    train_path = tmp_path / f"train_{epoch_idx:04d}.csv"
    assert train_path.exists()
    tlines = train_path.read_text().splitlines()
    assert tlines == ["epoch,batch,loss"]


def test_blocks_no_grad_file_for_nonzero_rank(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Avoid writing files for nonzero rank even when blocks are set.

    Verifies that with ``rank != 0`` the training, validation, and gradient
    record files are not created. The validation progress message is still
    printed to stdout.
    """
    device = torch.device("cpu")
    model = _DDPishWrapper()
    optimizer: torch.optim.Optimizer = torch.optim.SGD([torch.zeros(1)], lr=0.1)
    loss_fn: torch.nn.Module = torch.nn.MSELoss()
    scheduler = _DummyScheduler()

    num_train_batches = 1
    num_val_batches = 1
    epoch_idx = 8
    train_per_val = 1
    train_template = str(tmp_path / "train_<epochIDX>.csv")
    val_template = str(tmp_path / "val_<epochIDX>.csv")

    def dummy_train_step(
        _data: Iterable[object],
        mdl: _DDPishWrapper,
        _opt: torch.optim.Optimizer,
        _lfn: torch.nn.Module,
        dev: torch.device,
        _rnk: int,
        _ws: int,
    ) -> tuple[None, None, torch.Tensor]:
        """Set grads and return a single-element loss tensor."""
        mdl.module.p1.grad = torch.tensor([1.0, 2.0], device=dev)
        mdl.module.p2.grad = torch.tensor([5.0, 0.0, 0.0], device=dev)
        return None, None, torch.tensor([0.222], device=dev)

    def dummy_eval_step(
        _data: Iterable[object],
        mdl: _DDPishWrapper,
        _lfn: torch.nn.Module,
        dev: torch.device,
        _rnk: int,
        _ws: int,
    ) -> tuple[None, None, torch.Tensor]:
        """Return a single-element validation loss tensor."""
        return None, None, torch.tensor([0.333], device=dev)

    monkeypatch.setattr(
        lsc_policy_module,
        "train_lsc_policy_datastep",
        dummy_train_step,
    )
    monkeypatch.setattr(
        lsc_policy_module,
        "eval_lsc_policy_datastep",
        dummy_eval_step,
    )

    lsc_policy_module.train_lsc_policy_epoch(
        training_data=[1],
        validation_data=[1],
        num_train_batches=num_train_batches,
        num_val_batches=num_val_batches,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        LRsched=scheduler,
        epochIDX=epoch_idx,
        train_per_val=train_per_val,
        train_rcrd_filename=train_template,
        val_rcrd_filename=val_template,
        device=device,
        rank=1,
        world_size=2,
        blocks=_make_blocks(),
    )

    assert not (tmp_path / f"train_{epoch_idx:04d}.csv").exists()
    assert not (tmp_path / f"train_{epoch_idx:04d}_grad.csv").exists()
    assert not (tmp_path / f"val_{epoch_idx:04d}.csv").exists()

    captured = capsys.readouterr()
    assert f"Validating... {epoch_idx}" in captured.out
