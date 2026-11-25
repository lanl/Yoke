"""Tests for the per-channel LpLoss (UMich WAMRViT port)."""

import pytest
import torch

from yoke.losses.lp_loss import LpLoss, VectorLpLoss
from yoke.utils.training.datastep.array_output import train_array_datastep


@pytest.fixture
def preds_targets() -> tuple[torch.Tensor, torch.Tensor]:
    """Return a deterministic ``(y_pred, y)`` pair of shape (B, C, H, W)."""
    torch.manual_seed(0)
    B, C, H, W = 3, 4, 6, 5
    return torch.randn(B, C, H, W), torch.randn(B, C, H, W)


def _reference_per_bc(y_pred: torch.Tensor, y: torch.Tensor, eps: float) -> torch.Tensor:
    """Reference per-(B, C) absolute L2 norm matching the UMich loss."""
    B, C = y_pred.shape[0], y_pred.shape[1]
    diff = (y_pred - y).reshape(B, C, -1)
    return torch.sqrt((diff**2).sum(dim=-1) + eps)


def test_none_reduction_matches_reference(
    preds_targets: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """``reduction='none'`` returns the per-(B, C) L2 norm with no 1/M."""
    y_pred, y = preds_targets
    eps = 1e-4
    loss = LpLoss(d=2, p=2, method="abs", eps=eps, reduction="none")
    out = loss(y_pred, y)
    ref = _reference_per_bc(y_pred, y, eps)
    assert out.shape == (y_pred.shape[0], y_pred.shape[1])
    assert torch.allclose(out, ref)


def test_channel_mean_reduction(
    preds_targets: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """``reduction='channel_mean'`` returns per-sample mean over channels."""
    y_pred, y = preds_targets
    eps = 1e-4
    out = LpLoss(d=2, eps=eps, reduction="channel_mean")(y_pred, y)
    ref = _reference_per_bc(y_pred, y, eps).mean(dim=1)
    assert out.shape == (y_pred.shape[0],)
    assert torch.allclose(out, ref)


def test_mean_reduction_matches_reference(
    preds_targets: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """``reduction='mean'`` is the mean over batch and channel."""
    y_pred, y = preds_targets
    eps = 1e-4
    out = LpLoss(d=2, eps=eps, reduction="mean")(y_pred, y)
    ref = _reference_per_bc(y_pred, y, eps).mean()
    assert out.shape == ()
    assert torch.allclose(out, ref)


def test_mean_equals_channel_mean_batch_mean(
    preds_targets: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """The backprop scalar equals the batch-mean of recorded per-sample losses."""
    y_pred, y = preds_targets
    eps = 1e-4
    scalar = LpLoss(d=2, eps=eps, reduction="mean")(y_pred, y)
    per_sample = LpLoss(d=2, eps=eps, reduction="channel_mean")(y_pred, y)
    assert torch.allclose(scalar, per_sample.mean())


def test_sum_reduction(preds_targets: tuple[torch.Tensor, torch.Tensor]) -> None:
    """``reduction='sum'`` sums the per-(B, C) norm."""
    y_pred, y = preds_targets
    eps = 1e-4
    out = LpLoss(d=2, eps=eps, reduction="sum")(y_pred, y)
    ref = _reference_per_bc(y_pred, y, eps).sum()
    assert torch.allclose(out, ref)


def test_zero_error_gives_sqrt_eps() -> None:
    """A perfect prediction yields ``sqrt(eps)`` per channel, not exactly 0."""
    eps = 1e-4
    y = torch.randn(2, 3, 4, 4)
    out = LpLoss(d=2, eps=eps, reduction="none")(y.clone(), y)
    expected = torch.full((2, 3), float(eps) ** 0.5)
    assert torch.allclose(out, expected, atol=1e-7)


def test_relative_method(preds_targets: tuple[torch.Tensor, torch.Tensor]) -> None:
    """``method='rel'`` divides the per-channel norm by ``||y||``."""
    y_pred, y = preds_targets
    eps = 1e-4
    out = LpLoss(d=2, method="rel", eps=eps, reduction="none")(y_pred, y)
    diff_norm = _reference_per_bc(y_pred, y, eps)
    B, C = y.shape[0], y.shape[1]
    y_norm = torch.sqrt((y.reshape(B, C, -1) ** 2).sum(dim=-1) + eps)
    y_norm = torch.clamp(y_norm, min=eps)
    assert torch.allclose(out, diff_norm / y_norm)


def test_gradient_is_normalized_by_error_norm(
    preds_targets: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """The per-channel sqrt gives gradient ``diff / ||diff||`` for one term."""
    y_pred, y = preds_targets
    eps = 1e-4
    y_pred = y_pred.clone().requires_grad_(True)
    out = LpLoss(d=2, eps=eps, reduction="none")(y_pred, y)
    out[0, 0].backward()

    diff = (y_pred.detach()[0, 0] - y[0, 0]).reshape(-1)
    expected = diff / torch.sqrt((diff**2).sum() + eps)
    assert torch.allclose(y_pred.grad[0, 0].reshape(-1), expected, atol=1e-6)
    # Only the (0, 0) field receives gradient from a single-term backward.
    assert torch.all(y_pred.grad[0, 1] == 0.0)


def test_d3_norms_over_three_trailing_dims() -> None:
    """``d=3`` reduces over the trailing (T, H, W) dims (reference layout)."""
    eps = 1e-4
    B, C, T, H, W = 2, 3, 2, 4, 5
    y_pred = torch.randn(B, C, T, H, W)
    y = torch.randn(B, C, T, H, W)
    out = LpLoss(d=3, eps=eps, reduction="none")(y_pred, y)
    diff = (y_pred - y).reshape(B, C, -1)
    ref = torch.sqrt((diff**2).sum(dim=-1) + eps)
    assert out.shape == (B, C)
    assert torch.allclose(out, ref)


def test_p1_norm() -> None:
    """``p=1`` uses the general-p fallback (L1 norm, no 1/M)."""
    eps = 1e-4
    y_pred = torch.tensor([[[[1.0, -2.0], [3.0, -4.0]]]])
    y = torch.zeros_like(y_pred)
    out = LpLoss(d=2, p=1, eps=eps, reduction="none")(y_pred, y)
    assert torch.allclose(out, torch.tensor([[10.0]]))


def test_invalid_method_raises() -> None:
    """An unknown ``method`` raises ``ValueError``."""
    with pytest.raises(ValueError, match="method must be one of"):
        LpLoss(method="bogus")


def test_invalid_reduction_raises() -> None:
    """An unknown ``reduction`` raises ``ValueError``."""
    with pytest.raises(ValueError, match="reduction must be one of"):
        LpLoss(reduction="bogus")


def test_invalid_d_raises() -> None:
    """A non-positive ``d`` raises ``ValueError``."""
    with pytest.raises(ValueError, match="d must be a positive integer"):
        LpLoss(d=0)


# --- VectorLpLoss -----------------------------------------------------------


def test_vector_lp_hand_value() -> None:
    """[[3, 4]] against zeros gives sqrt(25 + eps), shape (1, 1)."""
    eps = 1e-4
    out = VectorLpLoss(eps=eps)(torch.tensor([[3.0, 4.0]]), torch.zeros(1, 2))
    assert out.shape == (1, 1)
    assert out.item() == pytest.approx((25.0 + eps) ** 0.5)


@pytest.mark.parametrize("method", ["abs", "rel"])
def test_vector_lp_matches_lploss(method: str) -> None:
    """Equals LpLoss(d=1) on the (B, 1, N) view, and reductions agree."""
    torch.manual_seed(1)
    y_pred, y = torch.randn(6, 4), torch.randn(6, 4)
    ref = LpLoss(d=1, method=method)(y_pred[:, None], y[:, None])
    out = VectorLpLoss(method=method)(y_pred, y)
    assert torch.allclose(out, ref)
    mean = VectorLpLoss(method=method, reduction="mean")(y_pred, y)
    total = VectorLpLoss(method=method, reduction="sum")(y_pred, y)
    assert mean.ndim == 0 and torch.allclose(mean, ref.mean())
    assert total.ndim == 0 and torch.allclose(total, ref.sum())


def test_vector_lp_finite_grad_at_zero_error() -> None:
    """The eps term keeps the gradient finite at zero error."""
    y = torch.randn(3, 4)
    y_pred = y.clone().requires_grad_(True)
    VectorLpLoss()(y_pred, y).sum().backward()
    assert torch.isfinite(y_pred.grad).all()


@pytest.mark.parametrize(
    ("a", "b"),
    [((4,), (4,)), ((2, 1, 4), (2, 1, 4)), ((2, 4), (2, 3))],
)
def test_vector_lp_bad_shapes(a: tuple[int, ...], b: tuple[int, ...]) -> None:
    """Non-2-D or mismatched inputs raise ValueError."""
    with pytest.raises(ValueError, match="2-D"):
        VectorLpLoss()(torch.zeros(a), torch.zeros(b))


def test_vector_lp_bad_reduction() -> None:
    """An unsupported reduction raises ValueError."""
    with pytest.raises(ValueError, match="reduction"):
        VectorLpLoss(reduction="channel_mean")


def test_vector_lp_through_array_datastep() -> None:
    """The array datastep records one norm per sample, shape (B,)."""
    torch.manual_seed(2)
    model = torch.nn.Linear(3, 4)
    opt = torch.optim.SGD(model.parameters(), lr=0.0)  # lr=0: weights fixed
    inpt, truth = torch.randn(5, 3), torch.randn(5, 4)
    loss_fn = VectorLpLoss()
    _, pred, per_sample = train_array_datastep(
        (inpt, truth), model, opt, loss_fn, torch.device("cpu")
    )
    assert per_sample.shape == (5,)
    assert torch.allclose(per_sample, loss_fn(pred, truth)[:, 0])
