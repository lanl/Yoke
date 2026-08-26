"""Exponential moving average (Polyak averaging) of selected model parameters.

A small, framework-light EMA shadow for a *subset* of a model's parameters --
here the trainable conditioner + output-head of the scalar-temporal LodeRunner
wrapper, with the frozen backbone deliberately excluded. Keeping the shadow keyed
by parameter NAME (rather than by position) makes it robust to reload: the
state_dict round-trips through the training checkpoint so the average survives the
per-epoch process restart used by the DDP harness (``cycle_epochs=1``), and the
eval scripts can overlay it onto a freshly-constructed model without needing an
optimizer.

Typical use in training::

    ema = ParamEMA(
        ((n, p) for n, p in model.named_parameters() if p.requires_grad),
        decay=0.999,
    )
    ...
    optimizer.step()
    ema.update((n, p) for n, p in model.named_parameters() if p.requires_grad)
    ...
    torch.save({..., "ema_state_dict": ema.state_dict()}, path)

and in eval to compare against the raw weights::

    ema = ParamEMA(named_trainable_params, decay=ckpt["ema_decay"])
    ema.load_state_dict(ckpt["ema_state_dict"])
    ema.copy_to(model.named_parameters())  # partial overlay of the shadowed subset
"""

from collections.abc import Iterable

import torch


class ParamEMA:
    """Name-keyed exponential moving average over a subset of parameters.

    The shadow tracks ``shadow <- decay * shadow + (1 - decay) * param`` for each
    supplied (name, parameter) pair. Only the provided parameters are shadowed, so
    passing just the trainable params keeps the frozen backbone out of the average
    (and out of the checkpoint).
    """

    def __init__(
        self,
        named_params: Iterable[tuple[str, torch.Tensor]],
        decay: float = 0.999,
    ) -> None:
        """Initialize the shadow from the current parameter values.

        Args:
            named_params (Iterable[tuple[str, torch.Tensor]]): (name, parameter)
                pairs to shadow, e.g. the ``requires_grad`` subset of
                ``model.named_parameters()``. Consumed once (may be a generator).
            decay (float): EMA decay in [0, 1). Higher = slower/smoother. The shadow
                is initialized to a detached clone of each parameter.
        """
        self.decay = float(decay)
        self.shadow: dict[str, torch.Tensor] = {
            name: p.detach().clone() for name, p in named_params
        }

    @torch.no_grad()
    def update(self, named_params: Iterable[tuple[str, torch.Tensor]]) -> None:
        """Update the shadow toward the current parameter values.

        Args:
            named_params (Iterable[tuple[str, torch.Tensor]]): The same (name,
                parameter) pairs supplied at construction (order-independent; matched
                by name). Names not present in the shadow are ignored; missing shadow
                entries are skipped.
        """
        d = self.decay
        for name, p in named_params:
            s = self.shadow.get(name)
            if s is None:
                continue
            # shadow <- decay * shadow + (1 - decay) * param
            s.lerp_(p.detach().to(s.device, s.dtype), 1.0 - d)

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return the shadow as a CPU-resident name->tensor dict for checkpointing."""
        return {name: s.detach().cpu().clone() for name, s in self.shadow.items()}

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load a previously saved shadow, copying into the existing tensors in place.

        Args:
            state_dict (dict[str, torch.Tensor]): Mapping produced by
                :meth:`state_dict`. Only keys present in the current shadow are
                loaded; the copy preserves each shadow tensor's device/dtype.
        """
        for name, s in self.shadow.items():
            saved = state_dict.get(name)
            if saved is not None:
                s.copy_(saved.to(s.device, s.dtype))

    @torch.no_grad()
    def copy_to(self, named_params: Iterable[tuple[str, torch.Tensor]]) -> None:
        """Overlay the shadow onto live parameters (a partial, in-place assignment).

        Only parameters whose names are in the shadow are overwritten, so this is a
        PARTIAL overlay -- the frozen/backbone params of the target model are left
        untouched. Use this at eval time to swap in the averaged weights.

        Args:
            named_params (Iterable[tuple[str, torch.Tensor]]): (name, parameter) pairs
                of the target model, e.g. ``model.named_parameters()``.
        """
        for name, p in named_params:
            s = self.shadow.get(name)
            if s is not None:
                p.data.copy_(s.to(p.device, p.dtype))
