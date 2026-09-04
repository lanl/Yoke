"""Exponential-moving-average (EMA) utilities for LodeRunner training.

This module provides a self-contained, ``diffusers``-free implementation of a
Diffusers-style *warmup* EMA built on top of
:class:`torch.optim.swa_utils.AveragedModel`. It maintains a non-gradient shadow
copy of the trainable parameters that is updated after each optimizer step, with
a decay schedule that warms up according to the number of averaging steps.

Target behavior (matching the ArtIMich recipe):

- EMA is a non-gradient shadow copy of the trainable parameters, updated
  *after each optimizer step*.
- Diffusers warmup decay schedule::

      s = num_averaged
      decay = clamp(1 - (1 + s / inv_gamma) ** (-power), min_decay, max_decay)
      ema_p = ema_p * decay + model_p * (1 - decay)

- The update is only applied once the global optimizer-step counter exceeds
  ``ema_update_after_step`` (default 1000). The first call to
  :meth:`AveragedModel.update_parameters` simply copies the model
  (``num_averaged == 0`` before the ``multi_avg_fn`` runs), and subsequent calls
  begin averaging with ``s = 1, 2, ...``.
- The EMA weights are saved as the production checkpoint.
- Under DDP each rank maintains an identical EMA copy after the synchronized
  optimizer step; only rank 0 need write the checkpoint.

Typical usage::

    from yoke.utils.ema import build_ema_model, save_ema_checkpoint

    ema_model = build_ema_model(model, device=device)
    global_step = 0
    ...
    # inside the training loop, after optimizer.step() and LRsched.step():
    global_step += 1
    if global_step > ema_update_after_step:
        ema_model.update_parameters(model)
    ...
    # at end of training (rank 0):
    save_ema_checkpoint(ema_model, "ema_weights.pth")

"""

from collections.abc import Callable

import torch
from torch.optim.swa_utils import AveragedModel


def make_warmup_ema_fn(
    max_decay: float = 0.9999,
    inv_gamma: float = 1.0,
    power: float = 2.0 / 3.0,
    min_decay: float = 0.0,
) -> Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]:
    r"""Build a Diffusers-style warmup ``multi_avg_fn`` for ``AveragedModel``.

    The returned callable has the signature expected by
    :class:`torch.optim.swa_utils.AveragedModel`'s ``multi_avg_fn`` argument:
    ``fn(ema_params, model_params, num_averaged)`` where ``ema_params`` and
    ``model_params`` are lists of parameter tensors and ``num_averaged`` is the
    number of models already averaged (a scalar tensor). The EMA parameters are
    updated *in place*.

    The decay at averaging step ``s = num_averaged`` is:

    .. math::

        \beta(s) = \mathrm{clamp}\left(
            1 - (1 + s / \gamma)^{-p},\; \beta_{\min},\; \beta_{\max}
        \right)

    and the update is ``ema_p = ema_p * beta + model_p * (1 - beta)``.

    Args:
        max_decay (float): Maximum (asymptotic) decay :math:`\beta_{\max}`.
        inv_gamma (float): Inverse-gamma factor :math:`\gamma` controlling the
            warmup rate.
        power (float): Warmup power :math:`p`.
        min_decay (float): Minimum decay :math:`\beta_{\min}` (floor at early
            steps).

    Returns:
        Callable: A ``multi_avg_fn`` for :class:`AveragedModel`.

    """

    def warmup_ema_fn(
        ema_params: list[torch.Tensor],
        model_params: list[torch.Tensor],
        num_averaged: int | torch.Tensor,
    ) -> None:
        """In-place warmup-EMA update of ``ema_params`` toward ``model_params``."""
        decay = compute_warmup_decay(
            num_averaged=num_averaged,
            max_decay=max_decay,
            inv_gamma=inv_gamma,
            power=power,
            min_decay=min_decay,
        )
        one_minus_decay = 1.0 - decay

        # torch._foreach_* gives an efficient fused update across the parameter
        # lists, matching AveragedModel's internal convention.
        torch._foreach_mul_(ema_params, decay)
        torch._foreach_add_(ema_params, model_params, alpha=one_minus_decay)

    return warmup_ema_fn


def compute_warmup_decay(
    num_averaged: int | torch.Tensor,
    max_decay: float = 0.9999,
    inv_gamma: float = 1.0,
    power: float = 2.0 / 3.0,
    min_decay: float = 0.0,
) -> float:
    r"""Compute the Diffusers warmup decay :math:`\beta(s)`.

    Args:
        num_averaged (int | torch.Tensor): Number of models already averaged,
            ``s`` in the schedule.
        max_decay (float): Maximum (asymptotic) decay.
        inv_gamma (float): Inverse-gamma factor controlling warmup rate.
        power (float): Warmup power.
        min_decay (float): Minimum decay floor.

    Returns:
        float: The decay :math:`\beta(s)` clamped to
        ``[min_decay, max_decay]``.

    """
    if isinstance(num_averaged, torch.Tensor):
        step = float(num_averaged.item())
    else:
        step = float(num_averaged)

    value = 1.0 - (1.0 + step / inv_gamma) ** (-power)
    return float(min(max(value, min_decay), max_decay))


def build_ema_model(
    model: torch.nn.Module,
    max_decay: float = 0.9999,
    inv_gamma: float = 1.0,
    power: float = 2.0 / 3.0,
    min_decay: float = 0.0,
    device: torch.device | str | None = None,
    use_buffers: bool = False,
) -> AveragedModel:
    """Build an :class:`AveragedModel` with a warmup-EMA averaging function.

    ``use_buffers=False`` is correct for the LodeRunner backbone because it uses
    LayerNorm/RMSNorm rather than BatchNorm running statistics; the buffers are
    non-stateful.

    Args:
        model (torch.nn.Module): The trainable model to shadow. For DDP wrap the
            *underlying* module (``ddp_model.module``), not the DDP wrapper.
        max_decay (float): Maximum (asymptotic) decay.
        inv_gamma (float): Inverse-gamma factor controlling warmup rate.
        power (float): Warmup power.
        min_decay (float): Minimum decay floor.
        device (torch.device | str | None): Device for the EMA copy.
        use_buffers (bool): Whether to average module buffers as well.

    Returns:
        AveragedModel: The EMA model wrapping a copy of ``model``.

    """
    avg_fn = make_warmup_ema_fn(
        max_decay=max_decay,
        inv_gamma=inv_gamma,
        power=power,
        min_decay=min_decay,
    )
    ema_model = AveragedModel(
        model,
        device=device,
        multi_avg_fn=avg_fn,
        use_buffers=use_buffers,
    )
    return ema_model


def save_ema_checkpoint(ema_model: AveragedModel, path: str) -> None:
    """Save the EMA weights as a plain model ``state_dict``.

    The saved state dict loads cleanly into a fresh model instance of the same
    class (e.g. :class:`~yoke.models.vit.swin.bomberman.LodeRunnerViT`) via
    :func:`load_ema_into_model`.

    Args:
        ema_model (AveragedModel): The EMA model whose ``module`` weights to save.
        path (str): Destination checkpoint path.

    """
    torch.save(ema_model.module.state_dict(), path)


def load_ema_into_model(model: torch.nn.Module, path: str) -> torch.nn.Module:
    """Load EMA weights saved by :func:`save_ema_checkpoint` into ``model``.

    Args:
        model (torch.nn.Module): A freshly-constructed model to receive the
            EMA weights.
        path (str): Path to the EMA checkpoint written by
            :func:`save_ema_checkpoint`.

    Returns:
        torch.nn.Module: The same ``model`` with EMA weights loaded.

    """
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    return model


if __name__ == "__main__":
    # Minimal smoke test / schedule illustration.
    for s in [0, 1, 2, 5, 10, 100, 10000]:
        beta = compute_warmup_decay(s)
        print(f"s={s:>6d}  beta={beta:.8f}")

    toy = torch.nn.Linear(4, 4)
    ema = build_ema_model(toy)
    opt = torch.optim.SGD(toy.parameters(), lr=0.1)
    for _ in range(5):
        loss = toy(torch.randn(8, 4)).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        ema.update_parameters(toy)
    print("EMA num_averaged:", int(ema.n_averaged.item()))
