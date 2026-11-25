"""Tests for the shared builder helpers in ``yoke.utils.builders``."""

import argparse

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from yoke.utils.builders import (
    build_adamw,
    build_from_checkpoint,
    build_from_pretrained,
    checkpoint_name,
    compute_last_epoch,
    default_mse_loss,
    move_optimizer_state_to_device,
)


class _TinyNet(nn.Module):
    """Minimal model for exercising the builders."""

    def __init__(self) -> None:
        """Initialize a single linear layer."""
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward map."""
        return self.linear(x)


def test_build_adamw_uses_explicit_args() -> None:
    """Explicit lr/weight_decay/betas/eps are honored."""
    model = _TinyNet()
    args = argparse.Namespace()
    opt = build_adamw(model, args, lr=3e-4, weight_decay=0.05)

    assert isinstance(opt, torch.optim.AdamW)
    group = opt.param_groups[0]
    assert group["lr"] == 3e-4
    assert group["weight_decay"] == 0.05
    assert group["betas"] == (0.9, 0.999)
    assert group["eps"] == 1e-8


def test_build_adamw_falls_back_to_args() -> None:
    """lr/weight_decay are pulled from args when not passed explicitly."""
    model = _TinyNet()
    args = argparse.Namespace(init_learnrate=7e-4, weight_decay=0.02)
    opt = build_adamw(model, args)

    group = opt.param_groups[0]
    assert group["lr"] == 7e-4
    assert group["weight_decay"] == 0.02


def test_build_adamw_defaults_when_args_missing() -> None:
    """Missing args attributes fall back to sensible defaults."""
    model = _TinyNet()
    args = argparse.Namespace()
    opt = build_adamw(model, args)

    group = opt.param_groups[0]
    assert group["lr"] == 1e-4
    assert group["weight_decay"] == 0.0


def test_move_optimizer_state_to_device_cpu() -> None:
    """Optimizer state tensors are moved to the target device in place."""
    model = _TinyNet()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Populate optimizer state.
    out = model(torch.randn(4, 3))
    out.sum().backward()
    opt.step()

    move_optimizer_state_to_device(opt, "cpu")

    for state in opt.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                assert value.device.type == "cpu"


def test_compute_last_epoch_fresh_run() -> None:
    """A fresh run (starting_epoch=0) yields the -1 sentinel."""
    assert compute_last_epoch(0, steps_per_epoch=250) == -1


def test_compute_last_epoch_continuation() -> None:
    """Continuation multiplies steps_per_epoch by (starting_epoch - 1)."""
    assert compute_last_epoch(1, steps_per_epoch=250) == 0
    assert compute_last_epoch(3, steps_per_epoch=250) == 500


def test_default_mse_loss() -> None:
    """The default loss is an unreduced MSE."""
    loss_fn = default_mse_loss()
    assert isinstance(loss_fn, nn.MSELoss)
    assert loss_fn.reduction == "none"


def test_checkpoint_name_formatting() -> None:
    """Checkpoint names zero-pad the study (3) and epoch (4) indices."""
    assert checkpoint_name(7, 12) == "study007_modelState_epoch0012.pth"
    assert checkpoint_name(123, 4567) == "study123_modelState_epoch4567.pth"


def test_build_from_checkpoint_fresh() -> None:
    """A fresh run constructs the model and returns None for the optimizer."""

    def model_args_fn(args: argparse.Namespace) -> dict:
        return {}

    builder = build_from_checkpoint(_TinyNet, model_args_fn)
    args = argparse.Namespace(continuation=False, checkpoint=None)
    model, model_args, model_class, starting_epoch, optimizer = builder(
        args, torch.device("cpu")
    )

    assert isinstance(model, _TinyNet)
    assert model_args == {}
    assert model_class is _TinyNet
    assert starting_epoch == 0
    assert optimizer is None


def test_build_from_checkpoint_continuation(
    tmp_path: object, monkeypatch: object
) -> None:
    """On continuation the builder reloads via load_model_and_optimizer."""
    import yoke.utils.builders as builders_mod

    sentinel_model = _TinyNet()
    sentinel_opt = torch.optim.AdamW(sentinel_model.parameters(), lr=1e-4)

    captured: dict = {}

    def fake_load(
        filepath: str,
        optimizer_class: type,
        optimizer_kwargs: dict,
        available_models: dict,
        device: object,
    ) -> tuple[nn.Module, torch.optim.Optimizer, int]:
        captured["filepath"] = filepath
        captured["available_models"] = available_models
        captured["optimizer_kwargs"] = optimizer_kwargs
        return sentinel_model, sentinel_opt, 9

    monkeypatch.setattr(builders_mod, "load_model_and_optimizer", fake_load)

    builder = build_from_checkpoint(_TinyNet, lambda a: {"width": 3})
    args = argparse.Namespace(
        continuation=True, checkpoint="ck.pth", init_learnrate=2e-4
    )
    model, model_args, model_class, starting_epoch, optimizer = builder(
        args, torch.device("cpu")
    )

    assert model is sentinel_model
    assert optimizer is sentinel_opt
    assert starting_epoch == 9
    assert model_args == {"width": 3}
    assert captured["filepath"] == "ck.pth"
    assert captured["available_models"] == {"_TinyNet": _TinyNet}
    assert captured["optimizer_kwargs"]["lr"] == 2e-4


class _SizedNet(nn.Module):
    """Minimal model with an architecture argument, for pretrained starts."""

    def __init__(self, width: int, shape: tuple[int, int] = (1, 2)) -> None:
        """Initialize one linear layer of the given width."""
        super().__init__()
        self.shape = shape
        self.linear = nn.Linear(width, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward map."""
        return self.linear(x)


def _save_pretrained(
    path: Path,
    model_args: dict,
    *,
    epoch: int = 6,
    prefix: str = "",
    model_class: str = "_SizedNet",
) -> nn.Module:
    """Write a checkpoint in ``save_model_and_optimizer``'s layout.

    Args:
        path (Path): Where to write the ``.pth`` file.
        model_args (dict): ``model_args`` to record and build the model from.
        epoch (int): Epoch to record.
        prefix (str): Prefix added to every weight name (``"module."`` mimics a
            DDP-wrapped save).
        model_class (str): Model class name to record.

    Returns:
        nn.Module: The saved model, for comparing weights.
    """
    model = _SizedNet(**model_args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # One optimizer step so the saved optimizer state is non-empty, like a real
    # training checkpoint.
    model(torch.ones(1, model_args["width"])).sum().backward()
    optimizer.step()
    torch.save(
        {
            "epoch": epoch,
            "model_class": model_class,
            "model_args": model_args,
            "model_state_dict": {
                prefix + key: value for key, value in model.state_dict().items()
            },
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )
    return model


def _fresh_args(**kwargs: object) -> argparse.Namespace:
    """Return args for a fresh (non-continuation) run."""
    return argparse.Namespace(continuation=False, checkpoint=None, **kwargs)


def test_build_from_pretrained_loads_weights_only(tmp_path: Path) -> None:
    """A pretrained start restores weights, epoch 0, and no optimizer."""
    path = tmp_path / "pretrained_ema.pth"
    saved = _save_pretrained(path, {"width": 3, "shape": (4, 5)})
    builder = build_from_pretrained(_SizedNet, lambda a: {"width": 3, "shape": [4, 5]})

    model, model_args, model_class, starting_epoch, optimizer = builder(
        _fresh_args(pretrained_model=str(path)), torch.device("cpu")
    )

    assert isinstance(model, _SizedNet)
    assert model_class is _SizedNet
    # The checkpoint's own model_args are used, and a list/tuple difference is
    # not an architecture difference.
    assert model_args == {"width": 3, "shape": (4, 5)}
    assert starting_epoch == 0
    assert optimizer is None
    for key, value in saved.state_dict().items():
        assert torch.equal(model.state_dict()[key], value)


def test_build_from_pretrained_strips_ddp_prefix(tmp_path: Path) -> None:
    """Weights saved from a DDP-wrapped model still load."""
    path = tmp_path / "wrapped.pth"
    saved = _save_pretrained(path, {"width": 3}, prefix="module.")
    builder = build_from_pretrained(_SizedNet, lambda a: {"width": 3})

    model, *_ = builder(_fresh_args(pretrained_model=str(path)), torch.device("cpu"))

    assert torch.equal(model.linear.weight, saved.linear.weight)


def test_build_from_pretrained_rejects_other_model_class(tmp_path: Path) -> None:
    """A checkpoint of another model class raises."""
    path = tmp_path / "other.pth"
    _save_pretrained(path, {"width": 3}, model_class="LodeRunnerViT")
    builder = build_from_pretrained(_SizedNet, lambda a: {"width": 3})

    with pytest.raises(ValueError, match="holds a LodeRunnerViT"):
        builder(_fresh_args(pretrained_model=str(path)), torch.device("cpu"))


def test_build_from_pretrained_rejects_model_args_mismatch(tmp_path: Path) -> None:
    """Differing architecture arguments raise and name the keys."""
    path = tmp_path / "pretrained.pth"
    _save_pretrained(path, {"width": 3})
    builder = build_from_pretrained(_SizedNet, lambda a: {"width": 4})

    with pytest.raises(ValueError, match="for: width"):
        builder(_fresh_args(pretrained_model=str(path)), torch.device("cpu"))


def test_build_from_pretrained_without_checkpoint_builds_fresh() -> None:
    """Without a pretrained checkpoint the model is built from model_args_fn."""
    builder = build_from_pretrained(_SizedNet, lambda a: {"width": 5})

    model, model_args, _, starting_epoch, optimizer = builder(
        _fresh_args(pretrained_model=None), torch.device("cpu")
    )

    assert model.linear.in_features == 5
    assert model_args == {"width": 5}
    assert starting_epoch == 0
    assert optimizer is None


def test_build_from_pretrained_continuation_reloads_checkpoint(
    tmp_path: Path,
) -> None:
    """On continuation, args.checkpoint wins and the optimizer and epoch return."""
    pretrained = tmp_path / "pretrained.pth"
    _save_pretrained(pretrained, {"width": 3}, epoch=6)
    restart = tmp_path / "study001_modelState_epoch0002.pth"
    _save_pretrained(restart, {"width": 3}, epoch=2)
    builder = build_from_pretrained(_SizedNet, lambda a: {"width": 3})
    args = argparse.Namespace(
        continuation=True,
        checkpoint=str(restart),
        pretrained_model=str(pretrained),
        init_learnrate=2e-4,
    )

    model, _, _, starting_epoch, optimizer = builder(args, torch.device("cpu"))

    assert starting_epoch == 2
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.state  # restored, not freshly built
