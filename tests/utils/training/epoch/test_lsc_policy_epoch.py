"""Unit tests for the train_lsc_policy_epoch function."""

import pathlib

import pytest
import torch

import yoke.utils.training.epoch.lsc_policy as lsc_policy_module


def test_train_and_validation_loops_run_for_rank0(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test that training and validation loops run for rank 0 process.

    Args:
        tmp_path (pathlib.Path): Temporary directory fixture.
        monkeypatch (pytest.MonkeyPatch): Fixture to patch module functions.
        capsys (pytest.CaptureFixture[str]): Fixture to capture standard output.

    """
    training_data = [1, 2, 3]
    validation_data = ["a", "b", "c"]
    num_train_batches = 2
    num_val_batches = 2
    epoch_idx = 2
    train_per_val = 1
    train_file_template = str(tmp_path / "train_<epochIDX>.csv")
    val_file_template = str(tmp_path / "val_<epochIDX>.csv")
    device = torch.device("cpu")
    rank = 0
    world_size = 1

    def dummy_train_step(
        data: object,
        model: torch.nn.Module,
        optimizer: object,
        loss_fn: object,
        dev: torch.device,
        rnk: int,
        ws: int,
    ) -> tuple[None, None, torch.Tensor]:
        return None, None, torch.tensor([0.123], device=dev)

    def dummy_eval_step(
        data: object,
        model: torch.nn.Module,
        loss_fn: object,
        dev: torch.device,
        rnk: int,
        ws: int,
    ) -> tuple[None, None, torch.Tensor]:
        return None, None, torch.tensor([0.456], device=dev)

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

    class DummyModel(torch.nn.Module):
        """Minimal model stub tracking train/eval calls."""

        def __init__(self) -> None:
            super().__init__()
            self.mode = None

        def train(self) -> None:
            self.mode = "train"

        def eval(self) -> None:
            self.mode = "eval"

    model = DummyModel()
    optimizer = object()
    loss_fn = object()

    class DummyScheduler:
        """Stub scheduler tracking number of step calls."""

        def __init__(self) -> None:
            self.step_count = 0

        def step(self) -> None:
            self.step_count += 1

    scheduler = DummyScheduler()

    lsc_policy_module.train_lsc_policy_epoch(
        training_data,
        validation_data,
        num_train_batches,
        num_val_batches,
        model,
        optimizer,
        loss_fn,
        scheduler,
        epoch_idx,
        train_per_val,
        train_file_template,
        val_file_template,
        device,
        rank,
        world_size,
    )

    assert scheduler.step_count == num_train_batches
    assert model.mode == "eval"

    train_path = tmp_path / f"train_{epoch_idx:04d}.csv"
    train_lines = train_path.read_text().splitlines()

    # training file now has a header row
    assert len(train_lines) == num_train_batches + 1
    assert train_lines[0] == "epoch,batch,loss"
    assert train_lines[1].startswith(f"{epoch_idx}, 0, 0.12300000")
    assert train_lines[2].startswith(f"{epoch_idx}, 1, 0.12300000")

    captured = capsys.readouterr()
    assert f"Validating... {epoch_idx}" in captured.out

    val_path = tmp_path / f"val_{epoch_idx:04d}.csv"
    val_lines = val_path.read_text().splitlines()
    assert len(val_lines) == num_val_batches
    assert val_lines[0].startswith(f"{epoch_idx}, 0, 0.45600000")
    assert val_lines[1].startswith(f"{epoch_idx}, 1, 0.45600000")


def test_validation_skipped_when_not_divisible(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test that validation is skipped when epoch index not divisible.

    Args:
        tmp_path (pathlib.Path): Temporary directory fixture.
        monkeypatch (pytest.MonkeyPatch): Fixture to patch module functions.
        capsys (pytest.CaptureFixture[str]): Fixture to capture standard output.

    """
    training_data = [1]
    validation_data = [1]
    num_train_batches = 1
    num_val_batches = 1
    epoch_idx = 3
    train_per_val = 2
    train_file_template = str(tmp_path / "train_<epochIDX>.csv")
    val_file_template = str(tmp_path / "val_<epochIDX>.csv")
    device = torch.device("cpu")
    rank = 0
    world_size = 1

    def dummy_train_step(
        data: object,
        model: torch.nn.Module,
        optimizer: object,
        loss_fn: object,
        dev: torch.device,
        rnk: int,
        ws: int,
    ) -> tuple[None, None, torch.Tensor]:
        return None, None, torch.tensor([0.321], device=dev)

    monkeypatch.setattr(
        lsc_policy_module,
        "train_lsc_policy_datastep",
        dummy_train_step,
    )

    class DummyModel(torch.nn.Module):
        """Model stub tracking train calls."""

        def __init__(self) -> None:
            super().__init__()
            self.mode = None

        def train(self) -> None:
            self.mode = "train"

    model = DummyModel()
    optimizer = object()
    loss_fn = object()

    class DummyScheduler:
        """Stub scheduler."""

        def __init__(self) -> None:
            self.step_count = 0

        def step(self) -> None:
            self.step_count += 1

    scheduler = DummyScheduler()

    lsc_policy_module.train_lsc_policy_epoch(
        training_data,
        validation_data,
        num_train_batches,
        num_val_batches,
        model,
        optimizer,
        loss_fn,
        scheduler,
        epoch_idx,
        train_per_val,
        train_file_template,
        val_file_template,
        device,
        rank,
        world_size,
    )

    assert scheduler.step_count == 1
    assert model.mode == "train"

    train_path = tmp_path / f"train_{epoch_idx:04d}.csv"
    assert train_path.exists()
    train_lines = train_path.read_text().splitlines()

    # one header row + one data row
    assert len(train_lines) == 2
    assert train_lines[0] == "epoch,batch,loss"

    captured = capsys.readouterr()
    assert captured.out == ""

    val_path = tmp_path / f"val_{epoch_idx:04d}.csv"
    assert not val_path.exists()


def test_no_files_created_for_nonzero_rank(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test that no record files are created when rank is not zero.

    Args:
        tmp_path (pathlib.Path): Temporary directory fixture.
        monkeypatch (pytest.MonkeyPatch): Fixture to patch module functions.
        capsys (pytest.CaptureFixture[str]): Fixture to capture standard output.

    """
    training_data = [1]
    validation_data = [1]
    num_train_batches = 1
    num_val_batches = 1
    epoch_idx = 4
    train_per_val = 1
    train_file_template = str(tmp_path / "train_<epochIDX>.csv")
    val_file_template = str(tmp_path / "val_<epochIDX>.csv")
    device = torch.device("cpu")
    rank = 1
    world_size = 2

    def dummy_train_step(
        data: object,
        model: torch.nn.Module,
        optimizer: object,
        loss_fn: object,
        dev: torch.device,
        rnk: int,
        ws: int,
    ) -> tuple[None, None, torch.Tensor]:
        """Dummy train step returning fixed tensor."""
        return None, None, torch.tensor([0.999], device=dev)

    def dummy_eval_step(
        data: object,
        model: torch.nn.Module,
        loss_fn: object,
        dev: torch.device,
        rnk: int,
        ws: int,
    ) -> tuple[None, None, torch.Tensor]:
        """Dummy eval step returning fixed tensor."""
        return None, None, torch.tensor([0.888], device=dev)

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

    class DummyModel(torch.nn.Module):
        """Model stub ignoring train/eval calls."""

        def train(self) -> None:
            pass

        def eval(self) -> None:
            pass

    model = DummyModel()
    optimizer = object()
    loss_fn = object()

    class DummyScheduler:
        """Stub scheduler tracking calls."""

        def __init__(self) -> None:
            self.step_count = 0

        def step(self) -> None:
            self.step_count += 1

    scheduler = DummyScheduler()

    lsc_policy_module.train_lsc_policy_epoch(
        training_data,
        validation_data,
        num_train_batches,
        num_val_batches,
        model,
        optimizer,
        loss_fn,
        scheduler,
        epoch_idx,
        train_per_val,
        train_file_template,
        val_file_template,
        device,
        rank,
        world_size,
    )

    train_path = tmp_path / f"train_{epoch_idx:04d}.csv"
    val_path = tmp_path / f"val_{epoch_idx:04d}.csv"
    assert not train_path.exists()
    assert not val_path.exists()

    captured = capsys.readouterr()
    assert f"Validating... {epoch_idx}" in captured.out


def test_zero_batches_creates_empty_files(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that empty record files are created when batch counts are zero.

    Args:
        tmp_path (pathlib.Path): Temporary directory fixture.
        monkeypatch (pytest.MonkeyPatch): Fixture to patch module functions.

    """
    training_data = [1]
    validation_data = [1]
    num_train_batches = 0
    num_val_batches = 0
    epoch_idx = 5
    train_per_val = 1
    train_file_template = str(tmp_path / "train_<epochIDX>.csv")
    val_file_template = str(tmp_path / "val_<epochIDX>.csv")
    device = torch.device("cpu")
    rank = 0
    world_size = 1

    def dummy_train_step(
        data: object,
        model: torch.nn.Module,
        optimizer: object,
        loss_fn: object,
        dev: torch.device,
        rnk: int,
        ws: int,
    ) -> None:
        pytest.fail("train_lsc_policy_datastep should not be called")

    def dummy_eval_step(
        data: object,
        model: torch.nn.Module,
        loss_fn: object,
        dev: torch.device,
        rnk: int,
        ws: int,
    ) -> None:
        pytest.fail("eval_lsc_policy_datastep should not be called")

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

    class DummyModel(torch.nn.Module):
        """Model stub ignoring train/eval calls."""

        def train(self) -> None:
            pass

        def eval(self) -> None:
            pass

    model = DummyModel()
    optimizer = object()
    loss_fn = object()

    class DummyScheduler:
        """Stub scheduler that should not be stepped."""

        def step(self) -> None:
            pytest.fail("Scheduler.step should not be called")

    scheduler = DummyScheduler()

    lsc_policy_module.train_lsc_policy_epoch(
        training_data,
        validation_data,
        num_train_batches,
        num_val_batches,
        model,
        optimizer,
        loss_fn,
        scheduler,
        epoch_idx,
        train_per_val,
        train_file_template,
        val_file_template,
        device,
        rank,
        world_size,
    )

    train_path = tmp_path / f"train_{epoch_idx:04d}.csv"
    val_path = tmp_path / f"val_{epoch_idx:04d}.csv"

    # training file now contains only the header when there are zero train batches
    assert train_path.exists()
    train_lines = train_path.read_text().splitlines()
    assert train_lines == ["epoch,batch,loss"]

    assert val_path.exists()
    assert val_path.stat().st_size == 0
