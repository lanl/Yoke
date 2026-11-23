"""Tests for the SwiGLU vector-to-vector MLP."""

from pathlib import Path

import pytest
import torch
from torch import nn

from yoke.models.mlp import SwiGLU_mlp, model_args_for_state_dict
from yoke.utils.checkpointing import load_model_and_optimizer, save_model_and_optimizer


@pytest.mark.parametrize(
    ("input_dim", "output_dim", "hidden_dims"),
    [(4, 4, [64, 64, 64]), (6, 4, [16]), (3, 2, [8, 16, 16, 4])],
)
def test_output_shape(input_dim: int, output_dim: int, hidden_dims: list[int]) -> None:
    """Output is (B, output_dim) for several configurations."""
    model = SwiGLU_mlp(input_dim, output_dim, hidden_dims)
    assert model(torch.randn(5, input_dim)).shape == (5, output_dim)


def test_backward_finite() -> None:
    """Every parameter gets a finite gradient."""
    model = SwiGLU_mlp(4, 4, [8, 8])
    model(torch.randn(3, 4)).sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name


def test_residual_only_on_equal_widths() -> None:
    """Skips are on equal-width layers only and pass the input through."""
    model = SwiGLU_mlp(4, 2, [8, 8, 6], residual=True)
    assert [layer.residual for layer in model.layers] == [False, True, False]

    # With its SwiGLU projection zeroed, a residual layer is the identity.
    layer = model.layers[1]
    nn.init.zeros_(layer.act.proj.weight)
    nn.init.zeros_(layer.act.proj.bias)
    x = torch.randn(3, 8)
    assert torch.equal(layer(x), x)

    plain = SwiGLU_mlp(4, 2, [8, 8], residual=False)
    assert not any(layer.residual for layer in plain.layers)


def test_options_build_and_run() -> None:
    """layer_norm=False and dropout>0 build; eval() is deterministic."""
    model = SwiGLU_mlp(4, 4, [8, 8], layer_norm=False, dropout=0.5)
    assert isinstance(model.layers[0].norm, nn.Identity)
    model.eval()
    x = torch.randn(4, 4)
    assert torch.equal(model(x), model(x))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"hidden_dims": []}, "at least one"),
        ({"hidden_dims": [8, 0]}, ">= 1"),
        ({"input_dim": 0}, ">= 1"),
        ({"dropout": 1.0}, "dropout"),
        ({"dropout": -0.1}, "dropout"),
    ],
)
def test_bad_args(kwargs: dict, match: str) -> None:
    """Invalid constructor arguments raise ValueError."""
    with pytest.raises(ValueError, match=match):
        SwiGLU_mlp(**kwargs)


def test_default_parameter_count() -> None:
    """The default 4 -> [64, 64, 64] -> 4 config has 17,796 parameters."""
    model = SwiGLU_mlp()
    assert sum(p.numel() for p in model.parameters()) == 17796
    # The legacy input LayerNorm(4) adds 8 parameters.
    legacy = SwiGLU_mlp(input_layer_norm=True)
    assert sum(p.numel() for p in legacy.parameters()) == 17804


def test_input_not_layer_normalized_by_default() -> None:
    """Only hidden layers after the first are normalized by default."""
    model = SwiGLU_mlp(4, 4, [8, 8, 8])
    assert isinstance(model.layers[0].norm, nn.Identity)
    assert all(isinstance(layer.norm, nn.LayerNorm) for layer in model.layers[1:])


def test_default_distinguishes_affine_shifted_inputs() -> None:
    """Default model: f(x) != f(3x + 2); the legacy input norm makes them equal."""
    torch.manual_seed(0)
    x = torch.randn(5, 4)
    model = SwiGLU_mlp(4, 4, [16, 16]).eval()
    assert (model(x) - model(3 * x + 2)).abs().max() > 1e-3
    legacy = SwiGLU_mlp(4, 4, [16, 16], input_layer_norm=True).eval()
    assert torch.allclose(legacy(x), legacy(3 * x + 2), atol=1e-4)


def test_model_args_for_legacy_state_dict() -> None:
    """Old checkpoints (input norm, no flag) rebuild and load exactly."""
    legacy_args = {"input_dim": 4, "output_dim": 2, "hidden_dims": [8, 8]}
    legacy = SwiGLU_mlp(**legacy_args, input_layer_norm=True)
    state = legacy.state_dict()

    args = model_args_for_state_dict(legacy_args, state)
    assert args["input_layer_norm"] is True
    assert "input_layer_norm" not in legacy_args  # input not mutated
    rebuilt = SwiGLU_mlp(**args)
    rebuilt.load_state_dict(state)
    x = torch.randn(3, 4)
    assert torch.equal(rebuilt.eval()(x), legacy.eval()(x))

    # New-style args and state dicts pass through unchanged.
    new = SwiGLU_mlp(**legacy_args)
    assert model_args_for_state_dict(legacy_args, new.state_dict()) == legacy_args


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    """A model rebuilt from its saved model_args gives the same output."""
    model_args = {"input_dim": 4, "output_dim": 3, "hidden_dims": [8, 8]}
    model = SwiGLU_mlp(**model_args)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = str(tmp_path / "ckpt.pth")
    save_model_and_optimizer(model, opt, 1, path, SwiGLU_mlp, model_args)

    loaded, _, epoch = load_model_and_optimizer(
        path,
        torch.optim.AdamW,
        {"lr": 1e-3},
        {"SwiGLU_mlp": SwiGLU_mlp},
        device="cpu",
    )
    x = torch.randn(2, 4)
    model.eval()
    loaded.eval()
    assert epoch == 1
    assert torch.equal(model(x), loaded(x))
