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
