"""Test module for yoke.utils.training.epoch.scalar_output."""

from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import pytest

import yoke.utils.training.epoch.scalar_output as scalar_output
from yoke.utils.training.epoch.scalar_output import train_scalar_epoch


def test_train_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test training epoch writes only training records without validation.

    This test ensures that when epochIDX is not divisible by train_per_val,
    only the training CSV file is created and contains the expected records.

    Args:
        tmp_path (Path): Temporary directory for file creation.
        monkeypatch (pytest.MonkeyPatch): MonkeyPatch fixture for patching.
        capsys (pytest.CaptureFixture[str]): Fixture to capture stdout and stderr.

    Returns:
        None: None.
    """

    # Dummy train step returns two losses per batch.
    def dummy_train_scalar_datastep(
        data: list,
        model: nn.Module,
        optimizer: optim.Optimizer,
        loss_fn: nn.Module,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.tensor([1.0]),
            torch.tensor([2.0]),
            torch.tensor([0.5, 0.75]),
        )

    # Patch only the train step; eval should not be called.
    monkeypatch.setattr(
        scalar_output, "train_scalar_datastep", dummy_train_scalar_datastep
    )

    # Prepare fake data and training artifacts.
    training_data = [None, None]
    validation_data = [None]  # won't be used
    model = nn.Linear(1, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()
    epoch_idx = 1
    train_per_val = 2
    device = torch.device("cpu")

    train_pattern = str(tmp_path / "train_<epochIDX>.csv")
    val_pattern = str(tmp_path / "val_<epochIDX>.csv")

    # Run epoch (no validation expected).
    train_scalar_epoch(
        training_data,
        validation_data,
        model,
        optimizer,
        loss_fn,
        epoch_idx,
        train_per_val,
        train_pattern,
        val_pattern,
        device,
    )

    # Check training file exists and contents are correct.
    train_file = tmp_path / "train_0001.csv"
    assert train_file.exists(), "Training record file was not created."

    lines = train_file.read_text().splitlines()
    # 2 batches * 2 losses each = 4 lines
    assert len(lines) == 4

    expected_losses = [0.5, 0.75]
    for idx, line in enumerate(lines):
        parts = [p.strip() for p in line.split(",")]
        epoch_val = int(parts[0])
        batch_val = int(parts[1])
        loss_val = float(parts[2])
        expected_batch = (idx // len(expected_losses)) + 1
        expected_loss = expected_losses[idx % len(expected_losses)]

        assert epoch_val == epoch_idx
        assert batch_val == expected_batch
        assert loss_val == pytest.approx(expected_loss)

    # No validation run: no output and no val file.
    captured = capsys.readouterr()
    assert captured.out == ""
    assert not (tmp_path / "val_0001.csv").exists()


def test_train_and_validate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test training and validation epochs write records and print validation.

    This test ensures that when epochIDX is divisible by train_per_val,
    both training and validation CSV files are created with correct records,
    and that validation is printed to stdout.

    Args:
        tmp_path (Path): Temporary directory for file creation.
        monkeypatch (pytest.MonkeyPatch): MonkeyPatch fixture for patching.
        capsys (pytest.CaptureFixture[str]): Fixture to capture stdout and stderr.

    Returns:
        None: None.
    """

    # Dummy train step returns two losses per batch.
    def dummy_train_scalar_datastep(
        data: list,
        model: nn.Module,
        optimizer: optim.Optimizer,
        loss_fn: nn.Module,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.tensor([1.0]),
            torch.tensor([2.0]),
            torch.tensor([0.5, 0.75]),
        )

    # Dummy eval step returns two losses per batch.
    def dummy_eval_scalar_datastep(
        data: list,
        model: nn.Module,
        loss_fn: nn.Module,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.tensor([3.0]),
            torch.tensor([4.0]),
            torch.tensor([0.25, 0.75]),
        )

    monkeypatch.setattr(
        scalar_output, "train_scalar_datastep", dummy_train_scalar_datastep
    )
    monkeypatch.setattr(
        scalar_output, "eval_scalar_datastep", dummy_eval_scalar_datastep
    )

    training_data = [None, None]
    validation_data = [None, None]
    model = nn.Linear(1, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()
    epoch_idx = 2
    train_per_val = 2
    device = torch.device("cpu")

    train_pattern = str(tmp_path / "train_<epochIDX>.csv")
    val_pattern = str(tmp_path / "val_<epochIDX>.csv")

    # Run epoch (validation should occur).
    train_scalar_epoch(
        training_data,
        validation_data,
        model,
        optimizer,
        loss_fn,
        epoch_idx,
        train_per_val,
        train_pattern,
        val_pattern,
        device,
    )

    # Capture and check validation printout.
    captured = capsys.readouterr()
    assert "Validating... 2" in captured.out

    # Check training records.
    train_file = tmp_path / "train_0002.csv"
    assert train_file.exists()
    t_lines = train_file.read_text().splitlines()
    assert len(t_lines) == 4

    # Check validation records.
    val_file = tmp_path / "val_0002.csv"
    assert val_file.exists()
    v_lines = val_file.read_text().splitlines()
    assert len(v_lines) == 4

    # Verify some sample values.
    # For training: epoch 2, batches 1 and 2, losses .5 and .75.
    parts = [p.strip() for p in t_lines[0].split(",")]
    assert int(parts[0]) == epoch_idx
    assert int(parts[1]) == 1
    assert float(parts[2]) == pytest.approx(0.5)

    # For validation: epoch 2, first loss of first batch = .25
    parts = [p.strip() for p in v_lines[0].split(",")]
    assert int(parts[0]) == epoch_idx
    assert int(parts[1]) == 1
    assert float(parts[2]) == pytest.approx(0.25)
