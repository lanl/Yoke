"""Tests for tutorials/onnx_deployment/run_onnx_model.py."""

import runpy
import sys
import os
import types
from pathlib import Path
from collections.abc import Iterator, Sequence, Callable

import numpy as np
import pytest
import torch
import tqdm

import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data_utils


def test_run_script_with_dummy_data(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test running the MNIST evaluation script with dummy components."""
    # Avoid filesystem side‐effects
    monkeypatch.setattr(os, "makedirs", lambda *args, **kwargs: None)
    # Turn tqdm into a no‐op
    monkeypatch.setattr(tqdm, "tqdm", lambda x, desc=None: x)

    # Dummy MNIST dataset
    class DummyDataset:
        def __init__(
            self,
            data_dir: str,
            train: bool,
            download: bool,
            transform: Callable[[torch.Tensor], torch.Tensor],
        ) -> None:
            del data_dir, train, download, transform
            self.data = [
                (torch.zeros((1, 28, 28)), 0),
                (torch.zeros((1, 28, 28)), 1),
            ]

        def __len__(self) -> int:
            return len(self.data)

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
            return self.data[idx]

    monkeypatch.setattr(datasets, "MNIST", DummyDataset)
    monkeypatch.setattr(transforms, "Compose", lambda x: (lambda y: y))

    # DataLoader that yields all samples in one batch
    class DummyLoader:
        def __init__(
            self, dataset: Sequence[tuple[torch.Tensor, int]], **kwargs: object
        ) -> None:
            del kwargs
            self.dataset = dataset

        def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
            """Yield a single batch containing all dataset samples."""
            data_list = [d for d, _ in self.dataset]
            lbl_list = [label for _, label in self.dataset]
            yield torch.stack(data_list), torch.tensor(lbl_list)

    monkeypatch.setattr(data_utils, "DataLoader", DummyLoader)

    # Stub out onnx_module.evaluate to be perfect one‐hot
    class DummyOnnxModule:
        def __init__(self, path: str) -> None:
            del path

        def evaluate(
            self,
            data_np: np.ndarray,
            verbose: bool = False,
            check_model: bool = False,
        ) -> list[np.ndarray]:
            batch = data_np.shape[0]
            out = np.zeros((batch, 2), dtype=np.float32)
            for i in range(batch):
                out[i, i] = 1.0
            return [out]

    dummy_pkg = types.ModuleType("yoke")
    dummy_sub = types.ModuleType("yoke.torch_training_utils")
    dummy_sub.onnx_module = DummyOnnxModule
    monkeypatch.setitem(sys.modules, "yoke", dummy_pkg)
    monkeypatch.setitem(sys.modules, "yoke.torch_training_utils", dummy_sub)

    project_root = Path(__file__).resolve().parents[3]
    script_path = project_root / "tutorials" / "onnx_deployment" / "run_onnx_model.py"
    runpy.run_path(str(script_path), run_name="__main__")

    captured = capsys.readouterr().out
    assert "MNIST test accuracy: 100.00%" in captured
