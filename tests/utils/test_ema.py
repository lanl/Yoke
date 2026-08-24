"""Tests for the warmup-EMA utilities in :mod:`yoke.utils.ema`.

Covers the Diffusers warmup decay schedule, the ``AveragedModel`` copy-then-
average timing, end-to-end training that produces distinct EMA weights, the
save/load round-trip into a fresh model, and a continuation (save/restore of the
EMA state plus global step counter) cycle.
"""

import os
import tempfile

import pytest
import torch
import torch.nn as nn

from yoke.utils.ema import (
    build_ema_model,
    compute_warmup_decay,
    load_ema_into_model,
    make_warmup_ema_fn,
    save_ema_checkpoint,
)


class TinyNet(nn.Module):
    """Minimal model for EMA testing (LayerNorm, not BatchNorm)."""

    def __init__(self, dim: int = 8) -> None:
        """Initialization."""
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward map."""
        return self.norm(self.linear(x))


# ============================================================================
# Schedule tests (Section 2.6)
# ============================================================================


@pytest.mark.parametrize("s", [1, 2, 5, 10, 100])
def test_warmup_decay_matches_formula(s: int) -> None:
    """Decay matches min(0.9999, 1 - (1 + s)^(-2/3)) for the reference config."""
    beta = compute_warmup_decay(
        s, max_decay=0.9999, inv_gamma=1.0, power=2.0 / 3.0, min_decay=0.0
    )
    expected = min(0.9999, 1.0 - (1.0 + s) ** (-2.0 / 3.0))
    assert beta == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_warmup_decay_zero_step_is_zero() -> None:
    """At s=0 the decay is 0 (pure copy of model params)."""
    assert compute_warmup_decay(0) == pytest.approx(0.0)


def test_warmup_decay_clamped_to_max() -> None:
    """Decay is clamped to max_decay for very large step counts."""
    beta = compute_warmup_decay(10**9, max_decay=0.9999)
    assert beta == pytest.approx(0.9999)


def test_warmup_ema_fn_in_place_update() -> None:
    """The multi_avg_fn updates EMA params in place: ema*beta + model*(1-beta)."""
    fn = make_warmup_ema_fn(max_decay=0.9999, inv_gamma=1.0, power=2.0 / 3.0)
    ema_p = [torch.ones(3)]
    model_p = [torch.zeros(3)]
    num_averaged = 1
    beta = compute_warmup_decay(num_averaged)

    fn(ema_p, model_p, num_averaged)

    # ema = 1*beta + 0*(1-beta) = beta
    assert torch.allclose(ema_p[0], torch.full((3,), beta), atol=1e-7)


# ============================================================================
# Timing tests (Section 2.3)
# ============================================================================


def test_first_update_copies_model() -> None:
    """The first update_parameters call copies the model (num_averaged 0 -> 1)."""
    torch.manual_seed(0)
    model = TinyNet()
    ema = build_ema_model(model)

    assert int(ema.n_averaged.item()) == 0

    # First call copies the model weights exactly.
    ema.update_parameters(model)
    assert int(ema.n_averaged.item()) == 1

    for p_ema, p_model in zip(ema.module.parameters(), model.parameters()):
        assert torch.allclose(p_ema, p_model)


def test_second_update_uses_step_one() -> None:
    """The second update averages with s=1 (the first nonzero Diffusers decay)."""
    torch.manual_seed(0)
    model = TinyNet()
    ema = build_ema_model(model)

    # First call: copy.
    ema.update_parameters(model)
    ema_after_copy = [p.detach().clone() for p in ema.module.parameters()]

    # Mutate the model so the next average is observable.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)

    # Second call: s = num_averaged = 1 at the time the avg_fn runs.
    beta = compute_warmup_decay(1)
    ema.update_parameters(model)

    for p_ema, p_copy, p_model in zip(
        ema.module.parameters(), ema_after_copy, model.parameters()
    ):
        expected = p_copy * beta + p_model * (1.0 - beta)
        assert torch.allclose(p_ema, expected, atol=1e-6)


# ============================================================================
# End-to-end tests (Section 2.6)
# ============================================================================


def test_ema_differs_from_raw_after_training() -> None:
    """After several steps EMA params differ from the raw model params."""
    torch.manual_seed(0)
    model = TinyNet()
    ema = build_ema_model(model)
    opt = torch.optim.SGD(model.parameters(), lr=0.5)

    for _ in range(10):
        x = torch.randn(16, 8)
        loss = model(x).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        ema.update_parameters(model)

    differs = any(
        not torch.allclose(p_ema, p_model)
        for p_ema, p_model in zip(ema.module.parameters(), model.parameters())
    )
    assert differs


def test_ema_save_load_roundtrip() -> None:
    """save_ema_checkpoint -> load_ema_into_model reproduces EMA weights."""
    torch.manual_seed(0)
    model = TinyNet()
    ema = build_ema_model(model)
    opt = torch.optim.SGD(model.parameters(), lr=0.5)

    for _ in range(5):
        loss = model(torch.randn(16, 8)).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        ema.update_parameters(model)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ema.pth")
        save_ema_checkpoint(ema, path)

        fresh = TinyNet()
        load_ema_into_model(fresh, path)

    for p_fresh, p_ema in zip(fresh.parameters(), ema.module.parameters()):
        assert torch.allclose(p_fresh, p_ema)


def test_ema_continuation_state_survives_restore() -> None:
    """EMA state_dict + global step counter survive a save/restore cycle."""
    torch.manual_seed(0)
    model = TinyNet()
    ema = build_ema_model(model)
    opt = torch.optim.SGD(model.parameters(), lr=0.5)

    global_step = 0
    for _ in range(7):
        loss = model(torch.randn(16, 8)).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        global_step += 1
        ema.update_parameters(model)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ema_state.pth")
        torch.save(
            {"ema_state_dict": ema.state_dict(), "global_step": global_step},
            path,
        )

        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        model2 = TinyNet()
        ema2 = build_ema_model(model2)
        ema2.load_state_dict(ckpt["ema_state_dict"])

    assert ckpt["global_step"] == global_step
    assert int(ema2.n_averaged.item()) == int(ema.n_averaged.item())
    for p2, p in zip(ema2.module.parameters(), ema.module.parameters()):
        assert torch.allclose(p2, p)
