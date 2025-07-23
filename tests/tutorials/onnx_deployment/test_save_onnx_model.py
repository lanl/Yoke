"""Tests for the ONNX deployment saving script."""

import os
import importlib
import sys
from pathlib import Path
from collections.abc import Iterator

import pytest
import torch
from torchvision import datasets


def test_save_onnx_model_calls_expected_routines(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test that save_onnx_model.py loads model and saves in ONNX format."""
    # Stub load_model_and_optimizer to capture its inputs and return dummy model
    import yoke.torch_training_utils as tr

    calls: dict = {}

    def fake_load_model_and_optimizer(
        path: str, optimizer: torch.optim.Optimizer, avail_models: dict, device: str
    ) -> tuple[torch.nn.Module, dict]:
        calls["path"] = path
        calls["optimizer"] = optimizer
        calls["avail_models"] = avail_models
        calls["device"] = device
        return torch.nn.Linear(1, 1), {"epoch": 0}

    monkeypatch.setattr(tr, "load_model_and_optimizer", fake_load_model_and_optimizer)

    # Stub onnx_module to capture init path and save arguments
    class DummyOnx:
        """Dummy ONNX handler recording save calls."""

        def __init__(self, path: str) -> None:
            calls["onnx_init_path"] = path

        def save(self, model: torch.nn.Module, example_input: torch.Tensor) -> None:
            calls["save_model"] = model
            calls["save_input_shape"] = example_input.shape

    monkeypatch.setattr(tr, "onnx_module", lambda path: DummyOnx(path))

    # Stub MNIST dataset to yield a batch of zeros
    class DummyDS:
        """Dummy dataset yielding one batch of zeros."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
            images = torch.zeros((2, 1, 28, 28))
            labels = torch.zeros((2,), dtype=torch.long)
            yield images, labels

    monkeypatch.setattr(datasets, "MNIST", DummyDS)

    # Stub DataLoader to simply return the dataset iterable
    import torch.utils.data as tud

    monkeypatch.setattr(tud, "DataLoader", lambda ds, **kwargs: ds)

    # Stub os.makedirs to record calls
    made: list = []
    monkeypatch.setattr(
        os, "makedirs", lambda p, exist_ok=True: made.append((p, exist_ok))
    )

    # Ensure fresh import of the script module
    module_name = "tutorials.onnx_deployment.save_onnx_model"
    if module_name in sys.modules:
        sys.modules.pop(module_name)

    # Act: import the script (executes top-level code)
    script_mod = importlib.import_module(module_name)

    # Assert load_model_and_optimizer was called with correct path and device
    expected_pth = script_mod.model_filepath + script_mod.model_name + ".pth"
    assert calls["path"] == expected_pth
    assert calls["device"] == "cpu"

    # Assert optimizer type and available_models dict identity
    assert isinstance(calls["optimizer"], torch.optim.Adadelta)
    assert calls["avail_models"] is script_mod.available_models

    # Assert that the ONNX output directory was created
    assert made == [(script_mod.onnx_model_savepath, True)]

    # Assert onnx_module initialized with correct path and save input shape
    expected_onx = script_mod.onnx_model_savepath + script_mod.onnx_model_name + ".onx"
    assert calls["onnx_init_path"] == expected_onx
    assert calls["save_input_shape"] == (1, 1, 28, 28)

    # Assert that the script printed the expected messages
    captured = capsys.readouterr()
    assert "Loading model from load_model_and_optimizer" in captured.out
    assert "Saving model in ONNX format" in captured.out
