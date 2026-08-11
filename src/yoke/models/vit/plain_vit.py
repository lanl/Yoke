"""Plain Vision-Transformer backbone with 2-D RoPE attention.

This module reimplements the *regular-grid, plain ViT* path of the UMich
WAMRViT ``QuadTreeTransformer`` in a self-contained, ``diffusers``-free form
suitable for use as a swappable backbone within Yoke's LodeRunner.

The backbone consumes and produces a token tensor of shape ``(B, N, D)``,
matching the interface of :class:`~yoke.models.vit.swin.unet.SwinUnetBackbone`.
It is *isotropic*: the token count ``N`` and embedding dimension ``D`` are held
constant through all transformer blocks (no patch merging/expansion).

Structural features carried over from the reference architecture:

- continuous 2-D rotary positional encoding from normalized patch centers,
- per-head RMSNorm on queries and keys prior to RoPE application,
- global scaled-dot-product attention over all spatial tokens,
- parallel attention + SwiGLU-MLP branches from a single pre-LayerNorm,
- concatenate-then-project (default) or sum fusion of the two branches,
- one residual update per block,
- a final non-affine LayerNorm.

"""

import torch
import torch.nn.functional as F
from torch import nn

from yoke.models.vit.rope import (
    SwiGLU,
    RotaryPositionalEmbeddingFromCenters,
    apply_rotary_emb,
    make_regular_centers,
)


class RopeAttention(nn.Module):
    r"""Global self-attention with per-head QK RMSNorm and 2-D RoPE.

    Projects the ``D``-dimensional token stream to queries, keys, and values
    (each ``heads * dim_head``), applies per-head RMSNorm to queries and keys,
    rotates queries and keys with the supplied ``(cos, sin)`` rotary embedding,
    and performs global scaled-dot-product attention across all tokens. The
    merged heads are projected back to ``D``.

    Args:
        query_dim (int): Token embedding dimension ``D``.
        heads (int): Number of attention heads.
        dim_head (int): Feature dimension per head. ``heads * dim_head`` is the
            inner attention dimension (equal to ``D`` in the isotropic ViT).
        dropout (float): Dropout probability on the output projection.
        bias (bool): Whether Q/K/V projections use a bias.
        out_bias (bool): Whether the output projection uses a bias.
        eps (float): Epsilon for the QK RMSNorm layers.

    """

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        out_bias: bool = True,
        eps: float = 1e-6,
    ) -> None:
        """Initialization for RopeAttention."""
        super().__init__()

        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head
        self.query_dim = query_dim
        self.dropout = dropout

        # Q/K/V projections
        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(query_dim, self.inner_dim, bias=bias)

        # Per-head RMSNorm on Q and K (normalizes the head_dim axis).
        self.norm_q = nn.RMSNorm(dim_head, eps=eps)
        self.norm_k = nn.RMSNorm(dim_head, eps=eps)

        # Output projection back to query_dim.
        self.to_out = nn.ModuleList(
            [
                nn.Linear(self.inner_dim, query_dim, bias=out_bias),
                nn.Dropout(dropout),
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Forward method for RopeAttention.

        Args:
            hidden_states (torch.Tensor): Tokens of shape ``(B, N, D)``.
            image_rotary_emb (tuple[torch.Tensor, torch.Tensor] | None):
                ``(cos, sin)`` rotary tensors each of shape ``(N, dim_head)``.

        Returns:
            torch.Tensor: Attention output of shape ``(B, N, D)``.

        """
        # Project and split into heads: (B, N, D) -> (B, N, H, D_head)
        query = self.to_q(hidden_states).unflatten(-1, (self.heads, self.dim_head))
        key = self.to_k(hidden_states).unflatten(-1, (self.heads, self.dim_head))
        value = self.to_v(hidden_states).unflatten(-1, (self.heads, self.dim_head))

        # Per-head RMSNorm on Q and K.
        query = self.norm_q(query)
        key = self.norm_k(key)

        # Apply rotary embedding to Q and K (sequence axis is dim 1).
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        # SDPA expects (B, H, N, D_head).
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
        )  # (B, H, N, D_head)

        # Merge heads back: (B, H, N, D_head) -> (B, N, D)
        hidden_states = hidden_states.transpose(1, 2).flatten(2)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)

        return hidden_states


class BasicTransformerBlock(nn.Module):
    r"""Parallel-branch transformer block with SwiGLU MLP and RoPE attention.

    Differs from a conventional pre-norm ViT block. From a single non-affine
    pre-LayerNorm, an attention branch and a SwiGLU-MLP branch run in parallel:

    .. math::

        z = \text{LN}(x) \\
        a = \text{RopeAttention}(z) \\
        m = \text{SwiGLU}(\text{Linear}(z)) \\
        x \leftarrow x + W_o [a \Vert m]

    where :math:`\Vert` is feature concatenation when ``concat=True`` (default).
    When ``concat=False`` the branches are each projected to ``D`` and summed.
    There is a single residual update per block.

    Args:
        num_attention_heads (int): Number of attention heads.
        attention_head_dim (int): Feature dimension per head. The hidden size is
            ``num_attention_heads * attention_head_dim``.
        mlp_ratio (float): MLP hidden width as a multiple of the hidden size.
        bias (bool): Whether attention projections use a bias.
        eps (float): Epsilon for LayerNorm and QK RMSNorm.
        concat (bool): If True, concatenate-then-project fusion; else sum fusion.

    """

    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 4.0,
        bias: bool = False,
        eps: float = 1e-6,
        concat: bool = True,
    ) -> None:
        """Initialization for BasicTransformerBlock."""
        super().__init__()

        hidden_size = num_attention_heads * attention_head_dim
        mlp_dim = int(hidden_size * mlp_ratio)

        self.attn = RopeAttention(
            query_dim=hidden_size,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=bias,
            out_bias=bias,
            eps=eps,
        )

        # Non-affine pre-LayerNorm shared by both branches.
        self.norm = nn.LayerNorm(hidden_size, eps=eps, elementwise_affine=False)

        # SwiGLU MLP branch.
        self.proj_mlp = nn.Linear(hidden_size, mlp_dim)
        self.act_mlp = SwiGLU(dim_in=mlp_dim, dim_out=mlp_dim)

        self.concat = concat
        if self.concat:
            # Fuse concatenated (attn || mlp) features back to hidden_size.
            self.proj_out = nn.Linear(hidden_size + mlp_dim, hidden_size)
        else:
            # Project each branch to hidden_size and sum.
            self.proj_mlp_down = nn.Linear(mlp_dim, hidden_size)
            self.proj_out = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Forward method for BasicTransformerBlock.

        Args:
            hidden_states (torch.Tensor): Tokens of shape ``(B, N, D)``.
            image_rotary_emb (tuple[torch.Tensor, torch.Tensor] | None):
                ``(cos, sin)`` rotary tensors each of shape ``(N, head_dim)``.

        Returns:
            torch.Tensor: Block output of shape ``(B, N, D)``.

        """
        norm_hidden_states = self.norm(hidden_states)  # (B, N, D)

        # SwiGLU MLP branch.
        mlp_hidden_states = self.proj_mlp(norm_hidden_states)  # (B, N, mlp_dim)
        mlp_hidden_states = self.act_mlp(mlp_hidden_states)  # (B, N, mlp_dim)

        # Attention branch (same normalized input).
        attn_output = self.attn(
            norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )  # (B, N, D)

        if self.concat:
            fused = torch.cat([attn_output, mlp_hidden_states], dim=2)
            fused = self.proj_out(fused)  # (B, N, D)
        else:
            mlp_hidden_states = self.proj_mlp_down(mlp_hidden_states)  # (B, N, D)
            fused = self.proj_out(attn_output) + mlp_hidden_states  # (B, N, D)

        return hidden_states + fused  # single residual update


class PlainViTBackbone(nn.Module):
    r"""Isotropic plain-ViT backbone operating on ``(B, N, D)`` tokens.

    A drop-in alternative to :class:`SwinUnetBackbone`. It expects a token
    tensor of shape ``(B, N, D)`` produced by an upstream patch embedding and
    returns ``(B, N, D)`` with the same token count and embedding dimension.

    The backbone owns a continuous 2-D RoPE built once from the regular-grid
    patch centers determined by ``patch_grid_size``, a stack of ``num_layers``
    :class:`BasicTransformerBlock` modules, and a final non-affine LayerNorm.

    .. note::
        The embedding dimension is fixed by ``num_attention_heads *
        attention_head_dim``. When wiring this into LodeRunner, ensure the
        upstream embedding dimension equals that product.

    Args:
        patch_grid_size (tuple[int, int]): Patch-grid ``(H_p, W_p)`` of the
            incoming tokens. ``N`` must equal ``H_p * W_p``.
        num_attention_heads (int): Number of attention heads per block.
        attention_head_dim (int): Feature dimension per head. The token
            embedding dimension is ``num_attention_heads * attention_head_dim``.
        num_layers (int): Number of transformer blocks.
        mlp_ratio (float): MLP hidden width as a multiple of the hidden size.
        rope_theta (float): Base period for the rotary frequency progression.
        rope_scale (tuple[float, float]): Per-axis coordinate scaling for the
            ``(x, y)`` patch centers.
        concat_mlp (bool): Concatenate-then-project (True) or sum (False) fusion.
        eps (float): Epsilon for LayerNorm and QK RMSNorm.
        bias (bool): Whether attention projections use a bias.
        verbose (bool): When True, prints derived dimensions at construction.

    """

    def __init__(
        self,
        patch_grid_size: tuple[int, int] = (112, 80),
        num_attention_heads: int = 8,
        attention_head_dim: int = 16,
        num_layers: int = 6,
        mlp_ratio: float = 4.0,
        rope_theta: float = 10000.0,
        rope_scale: tuple[float, float] = (1.0, 1.0),
        concat_mlp: bool = True,
        eps: float = 1e-6,
        bias: bool = False,
        verbose: bool = False,
    ) -> None:
        """Initialization for PlainViTBackbone."""
        super().__init__()

        self.patch_grid_size = patch_grid_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.num_layers = num_layers
        self.mlp_ratio = mlp_ratio

        self.emb_size = num_attention_heads * attention_head_dim
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        # The regular-grid path uses two rope axes (x, y), splitting head_dim.
        assert attention_head_dim % 2 == 0, (
            "attention_head_dim must be even so the 2-D RoPE can split it "
            "equally between the x and y axes."
        )
        rope_dim_list = [attention_head_dim // 2, attention_head_dim // 2]

        # Patch centers are fully determined by the patch grid; a 1x1 patch in
        # this normalized center space maps token index -> [x, y] in [0, 1].
        # We treat the incoming tokens as a full-grid patch embedding with unit
        # patch size in the normalized center coordinates.
        centers = make_regular_centers(
            image_size=patch_grid_size,
            patch_size=(1, 1),
        )  # (N, 2)
        self.register_buffer("centers", centers, persistent=False)

        self.rope = RotaryPositionalEmbeddingFromCenters(
            rope_dim_list=rope_dim_list,
            theta=rope_theta,
            scale=rope_scale,
        )

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    eps=eps,
                    bias=bias,
                    concat=concat_mlp,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = nn.LayerNorm(self.emb_size, eps=eps, elementwise_affine=False)

        if verbose:
            print("PlainViTBackbone patch-grid size:", patch_grid_size)
            print("PlainViTBackbone num_patches (N):", self.num_patches)
            print("PlainViTBackbone embedding dim (D):", self.emb_size)
            print("PlainViTBackbone rope_dim_list:", rope_dim_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method for PlainViTBackbone.

        Args:
            x (torch.Tensor): Token tensor of shape ``(B, N, D)`` with
                ``N == H_p * W_p`` and ``D == num_attention_heads *
                attention_head_dim``.

        Returns:
            torch.Tensor: Processed tokens of shape ``(B, N, D)``.

        """
        _, N, D = x.shape
        assert N == self.num_patches, (
            f"Token count {N} does not match patch grid {self.patch_grid_size} "
            f"(expected {self.num_patches})."
        )
        assert D == self.emb_size, (
            f"Token embedding dim {D} does not match backbone embedding dim "
            f"{self.emb_size} (= num_attention_heads * attention_head_dim)."
        )

        # Build rotary (cos, sin) once per forward in float32 for stability.
        with torch.autocast(x.device.type, enabled=False):
            rotary_emb = self.rope(self.centers)  # (cos, sin), each (N, head_dim)

        for block in self.transformer_blocks:
            x = block(x, image_rotary_emb=rotary_emb)

        x = self.norm_out(x)

        return x


if __name__ == "__main__":
    from yoke.utils.parameters import count_torch_params

    device = "cuda" if torch.cuda.is_available() else "cpu"

    patch_grid_size = (112, 80)  # matches (1120, 800) image with (10, 10) patch
    num_heads = 8
    head_dim = 16
    emb_size = num_heads * head_dim  # 128

    x = torch.rand(
        3, patch_grid_size[0] * patch_grid_size[1], emb_size, device=device
    )

    backbone = PlainViTBackbone(
        patch_grid_size=patch_grid_size,
        num_attention_heads=num_heads,
        attention_head_dim=head_dim,
        num_layers=6,
        mlp_ratio=4.0,
        concat_mlp=True,
        verbose=True,
    ).to(device)

    out = backbone(x)
    print("PlainViTBackbone input shape:", x.shape)
    print("PlainViTBackbone output shape:", out.shape)
    print("PlainViTBackbone output has NaNs:", torch.isnan(out).any().item())
    print("PlainViTBackbone parameters:", count_torch_params(backbone, trainable=True))

    # Sum-fusion variant
    backbone_sum = PlainViTBackbone(
        patch_grid_size=patch_grid_size,
        num_attention_heads=num_heads,
        attention_head_dim=head_dim,
        num_layers=6,
        concat_mlp=False,
    ).to(device)
    print("Sum-fusion output shape:", backbone_sum(x).shape)
