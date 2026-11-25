"""Multi-layer perceptrons for vector-to-vector regression.

:class:`SwiGLU_mlp` maps a feature vector ``(B, input_dim)`` to a target vector
``(B, output_dim)``. It fits Yoke's ``array`` batch contract (``model(inpt)``),
for example on tabular data loaded by
:class:`yoke.datasets.tabular_csv.TabularCSV_DataSet`.

Each hidden layer uses the gated SwiGLU activation from the LodeRunnerViT MLP
branch (:class:`yoke.models.vit.rope.SwiGLU`). Hidden layers after the first
are pre-LayerNorm (residual, where widths match) blocks, as in a transformer
MLP, so deeper variants stay easy to train. The **input** vector is not
layer-normalized by default (see ``input_layer_norm``).
"""

from collections.abc import Sequence

import torch
from torch import nn

from yoke.models.vit.rope import SwiGLU


class _SwiGLULayer(nn.Module):
    """One hidden layer: optional pre-LayerNorm, SwiGLU, dropout, and skip.

    Args:
        dim_in (int): Input width.
        dim_out (int): Output width.
        residual (bool): Add the input to the output. Only valid when
            ``dim_in == dim_out``; the caller decides.
        layer_norm (bool): Apply ``LayerNorm(dim_in)`` before the SwiGLU.
        dropout (float): Dropout probability after the SwiGLU.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        residual: bool,
        layer_norm: bool,
        dropout: float,
    ) -> None:
        """Build the layer."""
        super().__init__()
        self.residual = residual
        self.norm = nn.LayerNorm(dim_in) if layer_norm else nn.Identity()
        self.act = SwiGLU(dim_in=dim_in, dim_out=dim_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(B, dim_in)`` to ``(B, dim_out)``."""
        out = self.dropout(self.act(self.norm(x)))
        return x + out if self.residual else out


class SwiGLU_mlp(nn.Module):
    """MLP with SwiGLU hidden layers and a linear output head.

    Data flow, with ``h_0 = input_dim`` and ``h_i = hidden_dims[i-1]``:

    - hidden layer ``i`` (``i = 1..L``):
      ``[LayerNorm(h_{i-1})] -> SwiGLU(h_{i-1}, h_i) -> Dropout``, plus a skip
      connection when ``residual`` is set and ``h_{i-1} == h_i``. The
      LayerNorm is used when ``layer_norm`` is set, except on layer 1 (the raw
      input), which is normalized only if ``input_layer_norm`` is also set;
    - head: ``Linear(h_L, output_dim)``, with no activation, so the outputs can
      take any sign (for example z-scored targets).

    Shapes: ``(B, input_dim) -> (B, output_dim)``.

    Args:
        input_dim (int): Length of the input vector.
        output_dim (int): Length of the output vector.
        hidden_dims (Sequence[int]): Width of each hidden layer. Its length is
            the number of hidden layers.
        residual (bool): Use skip connections on equal-width hidden layers.
        layer_norm (bool): Apply a pre-LayerNorm in each hidden layer after the
            first.
        dropout (float): Dropout probability after each hidden activation.
        input_layer_norm (bool): Also apply LayerNorm to the raw input vector
            (layer 1; needs ``layer_norm``). Default ``False``. LayerNorm over
            the few features of a tabular input removes each row's mean and
            scale across its features, so the model cannot distinguish ``x``
            from ``a * x + b`` (``a > 0``), and it loses two degrees of freedom
            of the input. ``True`` reproduces models built before 2026-11-18,
            whose checkpoints carry a ``layers.0.norm`` parameter but no
            ``input_layer_norm`` entry in ``model_args`` (see
            :func:`model_args_for_state_dict`).

    Raises:
        ValueError: If a dimension is < 1, ``hidden_dims`` is empty, or
            ``dropout`` is not in ``[0, 1)``.
    """

    def __init__(
        self,
        input_dim: int = 4,
        output_dim: int = 4,
        hidden_dims: Sequence[int] = (64, 64, 64),
        residual: bool = True,
        layer_norm: bool = True,
        dropout: float = 0.0,
        input_layer_norm: bool = False,
    ) -> None:
        """Build the hidden layers and the output head."""
        super().__init__()
        hidden_dims = [int(h) for h in hidden_dims]
        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims must have at least one entry.")
        if min([input_dim, output_dim, *hidden_dims]) < 1:
            raise ValueError(
                "input_dim, output_dim, and hidden_dims must all be >= 1; got "
                f"{input_dim}, {output_dim}, {hidden_dims}."
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.input_layer_norm = input_layer_norm

        widths = [input_dim, *hidden_dims]
        self.layers = nn.ModuleList(
            _SwiGLULayer(
                dim_in=d_in,
                dim_out=d_out,
                # A skip connection needs matching widths, so the layers that
                # change width (e.g. the first, from input_dim) have none.
                residual=residual and d_in == d_out,
                # Never normalize the raw input unless asked (see docstring).
                layer_norm=layer_norm and (i > 0 or input_layer_norm),
                dropout=dropout,
            )
            for i, (d_in, d_out) in enumerate(zip(widths[:-1], widths[1:], strict=True))
        )
        self.head = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(B, input_dim)`` inputs to ``(B, output_dim)`` predictions.

        Args:
            x (torch.Tensor): Input batch, ``(B, input_dim)``.

        Returns:
            torch.Tensor: Predictions, ``(B, output_dim)``.
        """
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


def model_args_for_state_dict(model_args: dict, state_dict: dict) -> dict:
    """Return ``model_args`` that rebuild a checkpoint's ``SwiGLU_mlp`` exactly.

    Checkpoints written before ``input_layer_norm`` existed (2026-11-18) always
    normalized the input layer, and their ``model_args`` lack the key. For
    those, this sets ``input_layer_norm=True`` when the state dict holds the
    input-layer norm parameters, so ``load_state_dict`` matches. Newer
    ``model_args`` are returned unchanged.

    Args:
        model_args (dict): Saved constructor arguments.
        state_dict (dict): Saved model state dict.

    Returns:
        dict: A copy of ``model_args``, with ``input_layer_norm`` filled in for
        old checkpoints.
    """
    args = dict(model_args)
    if "input_layer_norm" not in args and "layers.0.norm.weight" in state_dict:
        args["input_layer_norm"] = True
    return args
