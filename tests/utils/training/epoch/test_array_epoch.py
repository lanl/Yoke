"""Tests for array_output epoch training and validation logic."""

from __future__ import annotations

from typing import Optional

from pathlib import Path

import pytest
import torch
from torch import nn

from yoke.utils.training.epoch import array_output as ao


def test_train_array_epoch_no_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test training epoch without validation."""
    # Prepare dummy data and filenames
    training_data = [1, 2]
    validation_data = [3]
    train_template = str(tmp_path / "train_<epochIDX>.csv")
    val_template = str(tmp_path / "val_<epochIDX>.csv")

    # Monkeypatch train_array_datastep to record calls and return fixed losses
    train_calls = []

    def dummy_train(
        databatch: object,
        model: object,
        optimizer: object,
        loss_fn: object,
        device: torch.device,
    ) -> tuple[None, None, torch.Tensor]:
        train_calls.append(databatch)
        return None, None, torch.tensor([0.1, 0.2])

    monkeypatch.setattr(ao, "train_array_datastep", dummy_train)
    # Monkeypatch eval_array_datastep to ensure it's not called
    monkeypatch.setattr(
        ao, "eval_array_datastep", lambda *args, **kwargs: (None, None, torch.tensor([]))
    )

    # Run with epochIDX that does not trigger validation
    model = object()
    optimizer = object()
    loss_fn = object()
    device = torch.device("cpu")
    ao.train_array_epoch(
        training_data,
        validation_data,
        model,
        optimizer,
        loss_fn,
        epochIDX=1,
        train_per_val=2,
        train_rcrd_filename=train_template,
        val_rcrd_filename=val_template,
        device=device,
    )

    # Ensure train_array_datastep was called for each training batch
    assert train_calls == [1, 2]

    # Check that the training record file was created and has correct content
    expected_train_file = tmp_path / "train_0001.csv"
    assert expected_train_file.exists()
    lines = expected_train_file.read_text().strip().splitlines()
    # Two losses per batch × two batches = 4 lines
    assert len(lines) == 4
    assert lines[0] == "1, 1, 0.10000000"
    assert lines[1] == "1, 1, 0.20000000"
    assert lines[2] == "1, 2, 0.10000000"
    assert lines[3] == "1, 2, 0.20000000"

    # Validation file should not exist for this epochIDX
    expected_val_file = tmp_path / "val_0001.csv"
    assert not expected_val_file.exists()


def test_train_array_epoch_with_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test training epoch with validation."""
    # Prepare dummy data and filenames
    training_data = [42]
    validation_data = ["a", "b"]
    train_template = str(tmp_path / "train_<epochIDX>.csv")
    val_template = str(tmp_path / "val_<epochIDX>.csv")

    # Monkeypatch train_array_datastep to return a single loss
    train_batches = []

    def dummy_train(
        databatch: object,
        model: object,
        optimizer: object,
        loss_fn: object,
        device: torch.device,
    ) -> tuple[None, None, torch.Tensor]:
        train_batches.append(databatch)
        return None, None, torch.tensor([0.3])

    monkeypatch.setattr(ao, "train_array_datastep", dummy_train)

    # Monkeypatch eval_array_datastep to return two losses per batch
    val_batches = []

    def dummy_eval(
        databatch: object, model: object, loss_fn: object, device: torch.device
    ) -> tuple[None, None, torch.Tensor]:
        val_batches.append(databatch)
        return None, None, torch.tensor([0.4, 0.5])

    monkeypatch.setattr(ao, "eval_array_datastep", dummy_eval)

    # Run with epochIDX that triggers validation (2 % 2 == 0)
    model = object()
    optimizer = object()
    loss_fn = object()
    device = torch.device("cpu")
    ao.train_array_epoch(
        training_data,
        validation_data,
        model,
        optimizer,
        loss_fn,
        epochIDX=2,
        train_per_val=2,
        train_rcrd_filename=train_template,
        val_rcrd_filename=val_template,
        device=device,
    )

    # Capture and verify validation printout
    captured = capsys.readouterr()
    assert "Validating... 2" in captured.out

    # Ensure train and eval functions were called expected number of times
    assert train_batches == [42]
    assert val_batches == ["a", "b"]

    # Check training record file
    expected_train_file = tmp_path / "train_0002.csv"
    assert expected_train_file.exists()
    train_lines = expected_train_file.read_text().strip().splitlines()
    assert len(train_lines) == 1

    # Allow for slight floating-point rounding differences
    epoch_str, batch_str, loss_str = train_lines[0].split(", ")
    assert epoch_str == "2"
    assert batch_str == "1"
    assert float(loss_str) == pytest.approx(0.3)

    # Check validation record file
    expected_val_file = tmp_path / "val_0002.csv"
    assert expected_val_file.exists()
    val_lines = expected_val_file.read_text().strip().splitlines()
    # Two losses per validation batch × two batches = 4 lines
    assert len(val_lines) == 4

    # Allow for slight floating-point rounding differences
    epoch_str, batch_str, loss_str = val_lines[0].split(", ")
    assert epoch_str == "2"
    assert batch_str == "1"
    assert float(loss_str) == pytest.approx(0.4)

    # Allow for slight floating-point rounding differences in validation losses
    expected = [
        (2, 1, 0.4),
        (2, 1, 0.5),
        (2, 2, 0.4),
        (2, 2, 0.5),
    ]
    for idx, (exp_epoch, exp_batch, exp_loss) in enumerate(expected):
        epoch_str, batch_str, loss_str = val_lines[idx].split(", ")
        assert int(epoch_str) == exp_epoch
        assert int(batch_str) == exp_batch
        assert float(loss_str) == pytest.approx(exp_loss)


def test_train_ddp_array_epoch_no_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test DDP training epoch without validation.

    This test stubs out the DDP datastep to return a fixed vector of
    per-sample losses and verifies that:

    * `train_DDP_array_epoch` iterates over the training data batches.
    * The per-sample losses are written to the correct train record file.
    * Validation is not triggered when `epochIDX % train_per_val != 0`.
    """
    # Prepare dummy data and filenames
    training_data = ["x", "y"]
    validation_data: list[object] = []
    train_template = str(tmp_path / "train_<epochIDX>.csv")
    val_template = str(tmp_path / "val_<epochIDX>.csv")

    # Record which training batches were seen
    train_batches: list[object] = []

    def dummy_train_ddp(
        databatch: object,
        model: object,
        optimizer: object,
        loss_fn: object,
        device: torch.device,
        rank: int,
        world_size: int,
    ) -> tuple[None, None, Optional[torch.Tensor]]:
        train_batches.append(databatch)
        # Simulate rank 0 gathering a final vector of per-sample losses.
        return None, None, torch.tensor([0.3, 0.7])

    # Monkeypatch DDP datasteps
    monkeypatch.setattr(ao, "train_DDP_array_datastep", dummy_train_ddp)
    monkeypatch.setattr(
        ao,
        "eval_DDP_array_datastep",
        lambda *args, **kwargs: (None, None, torch.tensor([])),
    )

    # Run a DDP epoch that does NOT trigger validation
    model = nn.Identity()
    optimizer = object()
    loss_fn = object()
    device = torch.device("cpu")

    class _NoOpSched:
        """Minimal LR scheduler stub for tests."""

        def step(self) -> None:
            """No-op step to satisfy interface."""
            return None

    ao.train_DDP_array_epoch(
        training_data,
        validation_data,
        num_train_batches=1,
        num_val_batches=2,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        LRsched=_NoOpSched(),
        epochIDX=1,
        train_per_val=3,
        train_rcrd_filename=train_template,
        val_rcrd_filename=val_template,
        device=device,
        rank=0,
        world_size=1,
    )

    # Check batches visited
    assert train_batches == training_data[:1]

    # Check training record file
    expected_train_file = tmp_path / "train_0001.csv"
    assert expected_train_file.exists()
    lines = expected_train_file.read_text().strip().splitlines()
    # Header + two per-sample losses from a single processed batch.
    assert len(lines) == 3
    assert lines[0] == "epoch,batch,loss"
    # First data line content (allow for float rounding)
    e1, b1, l1 = [s.strip() for s in lines[1].split(",")]
    assert e1 == "1"
    assert b1 == "0"  # batch indices are 0-based
    assert float(l1) == pytest.approx(0.3)
    # Second data line (optional extra check)
    e2, b2, l2 = [s.strip() for s in lines[2].split(",")]
    assert (e2, b2) == ("1", "0")
    assert float(l2) == pytest.approx(0.7)


def test_train_ddp_array_epoch_with_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Test DDP training epoch with validation triggered.

    This test stubs the DDP train/eval datasteps to return deterministic loss
    vectors, runs an epoch where `epochIDX % train_per_val == 0`, and checks:

    * The "Validating..." notice appears.
    * The training and validation files for the epoch are created.
    * Each file contains the expected number of lines and values.
    """
    # Prepare dummy data and filenames
    training_data = [101]
    validation_data = ["va", "vb"]
    train_template = str(tmp_path / "train_<epochIDX>.csv")
    val_template = str(tmp_path / "val_<epochIDX>.csv")

    # Record batches visited
    train_batches: list[object] = []
    val_batches: list[object] = []

    def dummy_train_ddp(
        databatch: object,
        model: object,
        optimizer: object,
        loss_fn: object,
        device: torch.device,
        rank: int,
        world_size: int,
    ) -> tuple[None, None, Optional[torch.Tensor]]:
        train_batches.append(databatch)
        return None, None, torch.tensor([0.25])

    def dummy_eval_ddp(
        databatch: object,
        model: object,
        loss_fn: object,
        device: torch.device,
        rank: int,
        world_size: int
    ) -> tuple[None, None, Optional[torch.Tensor]]:
        val_batches.append(databatch)
        return None, None, torch.tensor([0.4, 0.5])

    monkeypatch.setattr(ao, "train_DDP_array_datastep", dummy_train_ddp)
    monkeypatch.setattr(ao, "eval_DDP_array_datastep", dummy_eval_ddp)

    # Run a DDP epoch that DOES trigger validation (2 % 2 == 0)
    model = nn.Identity()
    optimizer = object()
    loss_fn = object()
    device = torch.device("cpu")

    class _NoOpSched:
        """Minimal LR scheduler stub for tests."""

        def step(self) -> None:
            """No-op step to satisfy interface."""
            return None

    ao.train_DDP_array_epoch(
        training_data,
        validation_data,
        num_train_batches=1,
        num_val_batches=2,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        LRsched=_NoOpSched(),
        epochIDX=2,
        train_per_val=2,
        train_rcrd_filename=train_template,
        val_rcrd_filename=val_template,
        device=device,
        rank=0,
        world_size=1,
    )

    # Verify notice and that both datasteps were invoked
    captured = capsys.readouterr()
    assert "Validating... 2" in captured.out
    assert train_batches == [101]
    assert val_batches == ["va", "vb"]

    # Training file: one batch × one loss = one line
    train_file = tmp_path / "train_0002.csv"
    assert train_file.exists()
    tlines = train_file.read_text().strip().splitlines()
    # Header + one per-sample loss from a single processed batch.
    assert len(tlines) == 2
    assert tlines[0] == "epoch,batch,loss"
    te, tb, tl = [s.strip() for s in tlines[1].split(",")]
    assert te == "2"
    assert tb == "0"  # 0-based batch index
    assert float(tl) == pytest.approx(0.25)

    # Validation file: two batches × two losses = four lines
    val_file = tmp_path / "val_0002.csv"
    assert val_file.exists()
    vlines = val_file.read_text().strip().splitlines()
    # No header for validation file, 4 data lines (2 batches × 2 per-sample losses)
    assert len(vlines) == 4
    # First validation data line sanity check
    ve, vb, vl = [s.strip() for s in vlines[0].split(",")]
    assert ve == "2"
    assert vb == "0"
    assert float(vl) == pytest.approx(0.4)
