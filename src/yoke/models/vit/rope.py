"""Rotary positional embedding primitives for the plain-ViT backbone.

This module provides self-contained (no ``diffusers`` dependency)
reimplementations of the transformer primitives used by the UMich WAMRViT
plain-ViT path:

- :class:`SwiGLU` gated MLP activation.
- :func:`get_1d_rotary_pos_embed` cos/sin generation.
- :func:`apply_rotary_emb` rotary application to queries/keys.
- :class:`RotaryPositionalEmbeddingFromCenters` continuous 2-D RoPE built from
  normalized patch-center coordinates.
- :func:`make_regular_centers` regular-grid patch-center generation.

The conventions here mirror the Hugging Face ``diffusers`` implementations
(``use_real=True`` with ``use_real_unbind_dim=-1``) so that the resulting
architecture matches the reference plain-ViT behavior.

"""

import torch
from torch import nn


class SwiGLU(nn.Module):
    r"""SwiGLU gated-linear activation.

    A gated activation for transformer MLP branches. Given input of dimension
    :math:`d_{in}`, an internal linear layer maps to :math:`2 d_{out}`, which is
    split into two halves :math:`(a, b)`. The output is:

    .. math::

        \text{SwiGLU}(x) = a \odot \text{SiLU}(b)

    where :math:`\text{SiLU}(z) = z \cdot \sigma(z)` and :math:`\odot` is
    element-wise multiplication. This matches the ``diffusers`` ``SwiGLU``
    module used by the reference architecture.

    Args:
        dim_in (int): Input feature dimension.
        dim_out (int): Output feature dimension.
        bias (bool): Whether the internal projection uses a bias. Default True.

    """

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True) -> None:
        """Initialization for SwiGLU."""
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2, bias=bias)
        self.activation = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward method for SwiGLU."""
        hidden_states = self.proj(hidden_states)
        hidden_states, gate = hidden_states.chunk(2, dim=-1)
        return hidden_states * self.activation(gate)


def get_1d_rotary_pos_embed(
    dim: int,
    pos: torch.Tensor,
    theta: float = 10000.0,
    ntk_factor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Generate 1-D rotary positional embeddings (cos, sin).

    Computes rotary frequencies for a set of (possibly continuous) positions.
    For a feature dimension :math:`d` (must be even), :math:`d/2` frequencies
    are computed as:

    .. math::

        \omega_i = \theta^{-2i/d}, \quad i = 0, 1, \dots, d/2 - 1

    The phase for position :math:`p` is the outer product :math:`p \cdot
    \omega_i`. Cosine and sine of this phase are each repeat-interleaved along
    the feature axis to width :math:`d`, matching the ``diffusers``
    convention (``use_real=True``).

    Args:
        dim (int): Feature dimension of the rotary embedding. Must be even.
        pos (torch.Tensor): 1-D tensor of positions, shape ``(N,)``.
        theta (float): Base period for the geometric frequency progression.
        ntk_factor (float): NTK scaling factor applied to ``theta``.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``(cos, sin)`` each of shape
        ``(N, dim)``.

    """
    assert dim % 2 == 0, "RoPE feature dimension must be even."

    theta = theta * ntk_factor

    # freqs: (dim//2,)
    freqs = 1.0 / (
        theta
        ** (torch.arange(0, dim, 2, dtype=torch.float32, device=pos.device) / dim)
    )

    pos = pos.to(torch.float32)

    # Outer product -> (N, dim//2)
    freqs = torch.outer(pos, freqs)

    # Repeat-interleave each frequency so consecutive feature pairs share a
    # rotation angle: (N, dim//2) -> (N, dim)
    cos = freqs.cos().repeat_interleave(2, dim=-1)
    sin = freqs.sin().repeat_interleave(2, dim=-1)

    return cos, sin


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: tuple[torch.Tensor, torch.Tensor],
    sequence_dim: int = 1,
) -> torch.Tensor:
    r"""Apply rotary positional embedding to a query or key tensor.

    The input ``x`` is expected in ``(B, N, H, D_head)`` layout (the default
    ``sequence_dim=1`` indicates the token/sequence axis). Treating consecutive
    feature pairs :math:`(x_1, x_2)` as a 2-D point, each pair is rotated by its
    per-position angle:

    .. math::

        x_1' = x_1 \cos - x_2 \sin, \quad x_2' = x_1 \sin + x_2 \cos

    This uses the ``diffusers`` ``use_real_unbind_dim=-1`` convention where the
    rotated-half is formed by interleaving ``[-x_2, x_1]``.

    Args:
        x (torch.Tensor): Query/key tensor of shape ``(B, N, H, D_head)``.
        freqs_cis (tuple[torch.Tensor, torch.Tensor]): ``(cos, sin)`` tensors
            each of shape ``(N, D_head)``.
        sequence_dim (int): Axis of ``x`` corresponding to the sequence/token
            dimension ``N``. Default 1.

    Returns:
        torch.Tensor: Rotated tensor with the same shape as ``x``.

    """
    cos, sin = freqs_cis  # each (N, D_head)

    # Reshape cos/sin to broadcast against (B, N, H, D_head).
    if sequence_dim == 1:
        # (N, D) -> (1, N, 1, D)
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
    elif sequence_dim == 2:
        # (N, D) -> (1, 1, N, D)
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
    else:
        raise ValueError(f"Unsupported sequence_dim={sequence_dim}, expected 1 or 2.")

    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)

    # Form the rotated-half via interleaved [-x2, x1] (use_real_unbind_dim=-1).
    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # each (..., D/2)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(-2)  # (..., D)

    return (x * cos) + (x_rotated * sin)


class RotaryPositionalEmbeddingFromCenters(nn.Module):
    r"""Continuous multi-axis RoPE built from center coordinates.

    Generates ``(cos, sin)`` rotary tensors for queries/keys from continuous
    per-token center coordinates. The head dimension is split among the axes
    according to ``rope_dim_list`` (which must sum to ``head_dim``). Each axis
    coordinate is scaled, passed through :func:`get_1d_rotary_pos_embed`, and
    the per-axis cos/sin segments are concatenated to fill ``head_dim``.

    For the regular plain-ViT path there are two axes ``(x, y)`` and
    ``rope_dim_list = (head_dim // 2, head_dim // 2)``.

    Args:
        rope_dim_list (list[int]): Per-axis feature dimensions. Must sum to the
            attention head dimension. Each entry must be even.
        theta (float | list[float]): Base period(s) for the frequency
            progression, either shared across axes or one per axis.
        ntk_factor (float): NTK scaling factor.
        scale (float | list[float] | None): Per-axis coordinate scaling. A
            single value broadcasts to all axes; ``None`` means unit scaling.

    """

    def __init__(
        self,
        rope_dim_list: list[int],
        theta: float | list[float] = 10000.0,
        ntk_factor: float = 1.0,
        scale: float | list[float] | None = None,
    ) -> None:
        """Initialization for RoPE-from-centers."""
        super().__init__()

        self.rope_dim_list = list(rope_dim_list)
        self.ntk_factor = ntk_factor
        self.num_segments = len(self.rope_dim_list)

        if isinstance(theta, (float, int)):
            self.theta = [float(theta)] * self.num_segments
        else:
            assert len(theta) == self.num_segments, (
                f"Length of theta ({len(theta)}) must match "
                f"len(rope_dim_list) ({self.num_segments})"
            )
            self.theta = list(theta)

        if scale is None:
            scale_vals = [1.0] * self.num_segments
        elif isinstance(scale, (float, int)):
            scale_vals = [float(scale)] * self.num_segments
        else:
            assert len(scale) == self.num_segments, (
                f"Length of scale ({len(scale)}) must match "
                f"len(rope_dim_list) ({self.num_segments})"
            )
            scale_vals = list(scale)

        self.register_buffer(
            "scale_tensor",
            torch.tensor(scale_vals, dtype=torch.float32),
            persistent=False,
        )

    @torch.no_grad()
    def forward(self, centers: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward method for RoPE-from-centers.

        Args:
            centers (torch.Tensor): Center coordinates of shape ``(N, D)`` or
                ``(B, N, D)`` where ``D == len(rope_dim_list)``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(cos, sin)`` each of shape
            ``(N, head_dim)``.

        """
        if centers.dim() == 3:
            centers_flat = centers.reshape(-1, centers.shape[-1])
        elif centers.dim() == 2:
            centers_flat = centers
        else:
            raise ValueError(
                f"centers must be [N, D] or [B, N, D], got {tuple(centers.shape)}"
            )

        if centers_flat.shape[-1] != self.num_segments:
            raise ValueError(
                f"Last dimension of centers ({centers_flat.shape[-1]}) must match "
                f"len(rope_dim_list) ({self.num_segments})"
            )

        device = centers_flat.device
        scales = self.scale_tensor.to(device)

        cos_segments = []
        sin_segments = []
        for i, dim in enumerate(self.rope_dim_list):
            coord = centers_flat[:, i] * scales[i]
            cos_i, sin_i = get_1d_rotary_pos_embed(
                dim,
                coord,
                theta=self.theta[i],
                ntk_factor=self.ntk_factor,
            )
            cos_segments.append(cos_i)
            sin_segments.append(sin_i)

        cos = torch.cat(cos_segments, dim=-1).to(device)
        sin = torch.cat(sin_segments, dim=-1).to(device)

        return cos, sin


def make_regular_centers(
    image_size: tuple[int, int],
    patch_size: tuple[int, int],
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    r"""Compute regular-grid patch-center coordinates for a plain ViT.

    Centers are normalized to ``[0, 1]`` relative to the full image domain. For
    a patch grid of :math:`H_p \times W_p`, the center of patch ``(row, col)``
    is:

    .. math::

        x = (col + 0.5) \cdot p_w / W, \quad y = (row + 0.5) \cdot p_h / H

    The returned tensor has columns ``[x_center, y_center]`` in row-major
    (``ij``) patch order, matching the token order produced by a flattened
    patch grid.

    Args:
        image_size (tuple[int, int]): Image ``(H, W)`` in pixels.
        patch_size (tuple[int, int]): Patch ``(p_h, p_w)`` in pixels.
        device (torch.device | str): Device for the returned tensor.

    Returns:
        torch.Tensor: Patch centers of shape ``(N, 2)`` with ``N = H_p * W_p``.

    """
    H, W = image_size
    p_h, p_w = patch_size
    H_p, W_p = H // p_h, W // p_w

    ys = (torch.arange(H_p, device=device) + 0.5) * p_h / H
    xs = (torch.arange(W_p, device=device) + 0.5) * p_w / W

    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    centers = torch.stack([xx, yy], dim=-1)  # (H_p, W_p, 2), columns [x, y]

    return centers.view(-1, 2)  # (N, 2)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # SwiGLU smoke test
    swiglu = SwiGLU(dim_in=32, dim_out=64).to(device)
    z = torch.rand(3, 10, 32, device=device)
    print("SwiGLU:", z.shape, "->", swiglu(z).shape)

    # Centers + RoPE smoke test
    image_size = (1120, 800)
    patch_size = (10, 10)
    centers = make_regular_centers(image_size, patch_size, device=device)
    print("centers:", centers.shape)  # (112*80, 2) = (8960, 2)

    head_dim = 64
    rope = RotaryPositionalEmbeddingFromCenters(
        rope_dim_list=[head_dim // 2, head_dim // 2],
        theta=10000.0,
        scale=(1.0, 1.0),
    ).to(device)
    cos, sin = rope(centers)
    print("cos/sin:", cos.shape, sin.shape)  # (8960, 64)

    # apply_rotary_emb smoke test: (B, N, H, D_head)
    B, N, H = 2, centers.shape[0], 8
    q = torch.rand(B, N, H, head_dim, device=device)
    q_rot = apply_rotary_emb(q, (cos, sin), sequence_dim=1)
    print("apply_rotary_emb:", q.shape, "->", q_rot.shape)
    assert q_rot.shape == q.shape
    print("rope.py smoke test passed.")
