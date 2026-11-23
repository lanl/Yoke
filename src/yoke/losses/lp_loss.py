"""Lp-norm loss matching the UMich WAMRViT ``LpLoss`` used for the PLI ViT.

This module ports the loss that the UMich WAMRViT PLI ViT (finest/mid)
experiments optimized (see
``misc_work_notes/ArtIMich/ArtIMICH_PLI_ViT_architecture_and_EMA_2ndDetails.md``
and the reference ``wamrvit/loss.py``). It differs from a plain
:class:`torch.nn.MSELoss` in two important ways:

1. **Per-field L2 norm, not per-element MSE.** For each ``(sample, channel)``
   the spatial (and optional temporal) error field is flattened and reduced via
   ``sqrt(sum(diff**2) + eps)`` -- an L2 *norm*, not a mean of squares. There is
   deliberately **no** ``1/M`` division by the number of field elements, mirroring
   the reference.

2. **Built-in per-channel gradient balancing.** Because each ``(sample,
   channel)`` term is a norm, its gradient with respect to the error field is
   ``diff / ||diff||`` -- unit-scaled per channel regardless of that channel's
   absolute error magnitude. This self-balances channels of very different
   physical scale, which is why the reference trains on raw (un-normalized)
   channels successfully.

:class:`VectorLpLoss` applies the same norm to ``(B, N)`` vector outputs (one
norm per sample), for tabular regression with the ``array`` datasteps.

The ``method="abs"`` form used by the finest PLI run is the absolute L2 norm.
The ``method="rel"`` form additionally divides by ``||y||`` per channel (a
relative error). The finest run uses ``abs``.

Reduction convention follows the reference (``reduce_dims=[0, 1]``,
``reductions="mean"`` -> mean over batch then channel). To keep the per-sample
loss-recording plumbing in the datastep functions in control of the final
reduction (as with ``nn.MSELoss(reduction="none")``), the default
``reduction="none"`` returns the per-``(B, C)`` norm tensor and lets the caller
reduce it.
"""

import torch
import torch.nn as nn


class LpLoss(nn.Module):
    r"""Per-channel Lp-norm loss (UMich WAMRViT ``LpLoss`` port).

    For prediction ``y_pred`` and target ``y`` of shape ``(B, C, *spatial)``,
    the per-``(sample, channel)`` loss flattens the trailing ``d`` dimensions and
    computes an Lp norm of the error:

    .. math::

        \ell_{b,c} = \left( \sum_{i} |e_{b,c,i}|^{p} \right)^{1/p}, \quad
        e = y_{\text{pred}} - y

    For ``p == 2`` a numerically stable form ``sqrt(sum(e**2) + eps)`` is used.
    When ``method == "rel"`` the per-channel norm is divided by ``||y||`` (with
    the denominator clamped to ``eps``) to give a relative error.

    .. note::
        There is intentionally no ``1/M`` averaging over the ``M`` flattened
        field elements -- this matches the reference and preserves the
        per-channel gradient-normalization property (gradient ``= e / ||e||``).

    Args:
        d (int): Number of trailing dimensions to reduce into the per-channel
            norm. For Yoke 2-frame predictions of shape ``(B, C, H, W)`` use
            ``d=2`` (norm over ``H, W``). The reference used ``d=3`` for its
            ``(B, C, T, H, W)`` tensors. Default ``2``.
        p (int): Order of the norm. Default ``2``.
        method (str): ``"abs"`` for an absolute norm, ``"rel"`` for a relative
            (target-normalized) norm. Default ``"abs"``.
        eps (float): Numerical-stability constant added inside the ``sqrt`` for
            ``p == 2`` and used to clamp the relative-norm denominator. Matches
            the reference finest value of ``1e-4``. Default ``1e-4``.
        reduction (str): Final reduction applied to the per-``(B, C)`` norm:

            - ``"none"`` (default): return the ``(B, C)`` tensor unreduced so a
              datastep can record per-sample losses (mirrors
              ``nn.MSELoss(reduction="none")`` usage).
            - ``"channel_mean"``: mean over the channel axis -> ``(B,)``
              per-sample loss.
            - ``"mean"``: mean over batch and channel -> scalar (matches the
              reference ``reduce_dims=[0, 1], reductions="mean"``).
            - ``"sum"``: sum over batch and channel -> scalar.

    """

    _VALID_METHODS = ("abs", "rel")
    _VALID_REDUCTIONS = ("none", "channel_mean", "mean", "sum")

    def __init__(
        self,
        d: int = 2,
        p: int = 2,
        method: str = "abs",
        eps: float = 1e-4,
        reduction: str = "none",
    ) -> None:
        """Initialize the LpLoss."""
        super().__init__()

        if method not in self._VALID_METHODS:
            raise ValueError(
                f"method must be one of {self._VALID_METHODS}; got {method!r}."
            )
        if reduction not in self._VALID_REDUCTIONS:
            raise ValueError(
                f"reduction must be one of {self._VALID_REDUCTIONS}; got {reduction!r}."
            )
        if d < 1:
            raise ValueError(f"d must be a positive integer; got {d}.")

        self.d = d
        self.p = p
        self.method = method
        self.eps = eps
        self.reduction = reduction

    def _per_channel_norm(self, flat: torch.Tensor) -> torch.Tensor:
        """Reduce a flattened error/target field to a per-``(B, C)`` norm.

        Args:
            flat (torch.Tensor): Tensor of shape ``(B, C, M)`` where ``M`` is the
                product of the flattened trailing ``d`` dimensions.

        Returns:
            torch.Tensor: Per-``(B, C)`` Lp norm of shape ``(B, C)``.
        """
        if self.p == 2:
            # Numerically stable L2 norm: sqrt(sum(x^2) + eps). No 1/M averaging.
            return torch.sqrt((flat**2).sum(dim=-1) + self.eps)
        # General p-norm fallback.
        return torch.norm(flat, p=self.p, dim=-1)

    def forward(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the per-channel Lp-norm loss.

        Args:
            y_pred (torch.Tensor): Predicted tensor of shape ``(B, C, *spatial)``.
            y (torch.Tensor): Target tensor of the same shape as ``y_pred``.

        Returns:
            torch.Tensor: The loss, reduced according to ``reduction``:
            ``(B, C)`` for ``"none"``, ``(B,)`` for ``"channel_mean"``, or a
            scalar for ``"mean"``/``"sum"``.
        """
        y = y.float()
        y_pred = y_pred.float()

        # Flatten the trailing ``d`` dims: (B, C, *spatial) -> (B, C, M).
        diff = torch.flatten(y_pred - y, start_dim=-self.d).flatten(
            start_dim=1, end_dim=-2
        )
        norm = self._per_channel_norm(diff)  # (B, C)

        if self.method == "rel":
            y_flat = torch.flatten(y, start_dim=-self.d).flatten(start_dim=1, end_dim=-2)
            y_norm = self._per_channel_norm(y_flat)  # (B, C)
            y_norm = torch.clamp(y_norm, min=self.eps)
            norm = norm / y_norm

        if self.reduction == "none":
            return norm
        if self.reduction == "channel_mean":
            return norm.mean(dim=1)
        if self.reduction == "sum":
            return norm.sum()
        # "mean": mean over batch and channel (reference convention).
        return norm.mean()


class VectorLpLoss(nn.Module):
    r"""Per-sample Lp norm of the error between two batches of vectors.

    For prediction and target vectors of shape ``(B, N)`` (for example the
    z-scored outputs of a tabular regression), the per-sample loss is the Lp
    norm of the error vector, computed by :class:`LpLoss` with ``d=1`` on a
    ``(B, 1, N)`` view:

    .. math::

        \ell_b = \left( \sum_{i=1}^{N} |\hat{y}_{b,i} - y_{b,i}|^{p} \right)^{1/p}

    For ``p == 2`` this is ``sqrt(sum(e**2) + eps)``. Its gradient
    ``e / ||e||`` has unit size for every sample, so, unlike MSE, samples with
    large errors do not dominate the update.

    The default ``reduction="none"`` returns shape ``(B, 1)``, which is what the
    ``array`` datasteps in :mod:`yoke.utils.training.datastep.array_output`
    need. They record the mean over the non-batch dims (here, the norm itself)
    and backpropagate the batch mean. A ``(B,)`` loss would not work there: its
    non-batch reduction is over no dims, which collapses it to a scalar.

    Args:
        p (int): Order of the norm. Default ``2``.
        method (str): ``"abs"`` for the absolute norm, ``"rel"`` to divide by
            the target's norm ``||y_b||`` (clamped to ``eps``). Default
            ``"abs"``.
        eps (float): Stability constant, as in :class:`LpLoss`. Default
            ``1e-4``.
        reduction (str): ``"none"`` gives ``(B, 1)``; ``"mean"`` or ``"sum"``
            over the batch gives a scalar.

    Raises:
        ValueError: On an unknown ``reduction`` or ``method``, or, in
            ``forward``, if the inputs are not 2-D tensors of equal shape.
    """

    _VALID_REDUCTIONS = ("none", "mean", "sum")

    def __init__(
        self,
        p: int = 2,
        method: str = "abs",
        eps: float = 1e-4,
        reduction: str = "none",
    ) -> None:
        """Initialize the VectorLpLoss."""
        super().__init__()
        if reduction not in self._VALID_REDUCTIONS:
            raise ValueError(
                f"reduction must be one of {self._VALID_REDUCTIONS}; got {reduction!r}."
            )
        self.reduction = reduction
        # The wrapped loss stays unreduced; the reduction is applied here so that
        # "none" keeps the (B, 1) shape (LpLoss has no such option).
        self.lp = LpLoss(d=1, p=p, method=method, eps=eps, reduction="none")

    def forward(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the per-sample vector Lp-norm loss.

        Args:
            y_pred (torch.Tensor): Predictions, ``(B, N)``.
            y (torch.Tensor): Targets, ``(B, N)``.

        Returns:
            torch.Tensor: ``(B, 1)`` for ``"none"``, otherwise a scalar.
        """
        if y_pred.ndim != 2 or y_pred.shape != y.shape:
            raise ValueError(
                "VectorLpLoss expects 2-D (B, N) inputs of equal shape; got "
                f"{tuple(y_pred.shape)} and {tuple(y.shape)}."
            )
        # Insert a singleton channel axis: (B, N) -> (B, 1, N) -> norm (B, 1).
        norm = self.lp(y_pred.unsqueeze(1), y.unsqueeze(1))
        if self.reduction == "none":
            return norm
        if self.reduction == "sum":
            return norm.sum()
        return norm.mean()
