"""Patch manipulation layers.

Includes *merging*, *expanding*, *depatching* layer definitions.

The goal of these layers is to reduce or increase the number of tokens of a
patch embedding while simultaneously modifying the embedding
dimension. Hypothetically this should allow a SWIN transformer to learn a
heirarchical feature representation for the image.

"""

import torch
from torch import nn
from einops import rearrange


class PatchMerge(nn.Module):
    r"""Merge patches and increase embedding dimension.

    This layer is passed a tensor of shape :math:`(B, L, C)`, i.e., batches of
    :math:`L` tokens, each of embedding size :math:`C`. The embedding size must
    match the token embedding dimension from the previous layer. It is assumed
    :math:`L = H \times W` with :math:`H` divisible by :math:`s_1` and :math:`W`
    divisible by :math:`s_2`.

    The tokens are reshaped into groups of shape:

    .. math::

        (H', s_1, W', s_2) \text{ with } H' = H / s_1 \text{ and } W' = W / s_2

    Then, the input is remapped:

    .. math::

        B \times (H' \cdot s_1 \cdot W' \cdot s_2) \times C
        \rightarrow
        B \times (H' \cdot W') \times (s_1 \cdot s_2 \cdot C)

    Finally, a linear embedding is applied to produce a tensor of shape:

    .. math::

        (B, H' \cdot W', \text{embedding factor} \cdot C)

    Args:
        emb_size (int): Incoming embedding dimension
        emb_factor (int): Up-scaling factor for embedding dimension
        patch_grid_size (Tuple[int, int]): Incoming patch-grid dimensions within
                                           the token set
        s1 (int): Height reduction factor for the patch grid
        s2 (int): Width reduction factor for the patch grid

    """

    def __init__(
        self,
        emb_size: int = 64,
        emb_factor: int = 2,
        patch_grid_size: (int, int) = (64, 64),
        s1: int = 2,
        s2: int = 2,
    ) -> None:
        """Initialization for Patch Merging."""
        super().__init__()
        # Check size compatibilities
        try:
            msg = "Patch-grid height not divisible by height of patch-merge scale!!!"
            assert patch_grid_size[0] % s1 == 0, msg
        except AssertionError as e:
            msg_tuple = (
                "Patch-grid height:",
                patch_grid_size[0],
                "Patch-merge scale height:",
                s1,
            )
            e.args += msg_tuple
            raise

        try:
            msg = "Patch-grid width not divisible by width of patch-merge scale!!!"
            assert patch_grid_size[1] % s2 == 0, msg
        except AssertionError as e:
            msg_tuple = (
                "Patch-grid width:",
                patch_grid_size[1],
                "Patch-merge scale width:",
                s2,
            )
            e.args += msg_tuple
            raise

        self.in_emb_size = emb_size

        # Patch grid parameters
        self.H = patch_grid_size[0]
        self.W = patch_grid_size[1]

        # Patch division factors
        self.s1 = s1
        self.s2 = s2

        # New patch grid
        self.out_patch_grid_size = (int(self.H / self.s1), int(self.W / self.s2))

        # Embedding dimension factor
        self.emb_factor = emb_factor
        self.out_emb_size = self.emb_factor * self.in_emb_size

        # Linear re-embedding
        self.linear = nn.Linear(self.s1 * self.s2 * self.in_emb_size, self.out_emb_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method for Patch Merging."""
        # The input tensor is shape (B, num_tokens, embedding_dim)
        _, L, _ = x.shape

        # NOTE: The number of tokens is assumed to be L=H*W
        assert L == self.H * self.W

        x = rearrange(
            x,
            "b (h s1 w s2) c -> b (h w) (s1 s2 c)",
            s1=self.s1,
            s2=self.s2,
            h=self.out_patch_grid_size[0],
            w=self.out_patch_grid_size[1],
        )
        x = self.linear(x)

        return x


class PatchExpand(nn.Module):
    r"""Expand patches and decrease embedding dimension.

    This layer receives a tensor of shape :math:`(B, L, C)`, representing
    batches of :math:`L` tokens, each with embedding dimension :math:`C`.
    The embedding size :math:`C` must match the output of the previous layer.

    It is assumed that :math:`L = H \times W`, where :math:`(H, W)` define
    the patch grid from the previous layer.

    The expansion process proceeds as follows:

    .. math::

        (B, H \cdot W, C)
        \xrightarrow{\text{linear}}
        (B, H \cdot W, n \cdot C)
        \xrightarrow{\text{rearrange}}
        (B, H \cdot s_1 \cdot W \cdot s_2, \frac{n \cdot C}{s_1 \cdot s_2})

    First, the embedding dimension is expanded linearly by a factor of
    :math:`n`. Then, the embedding is rearranged by distributing the added
    capacity into more tokens, increasing the patch count while reducing the
    per-token dimension accordingly.

    .. note::
        :math:`n \cdot C` must be divisible by :math:`s_1 \cdot s_2`

    Args:
        emb_size (int): Incoming embedding dimension
        emb_factor (int): Up-scaling factor for embedding dimension
        patch_grid_size (Tuple[int, int]): Incoming patch-grid dimensions within
                                           the token set
        s1 (int): Height scaling factor for the patch grid
        s2 (int): Width scaling factor for the patch grid


    """

    def __init__(
        self,
        emb_size: int = 64,
        emb_factor: int = 2,
        patch_grid_size: (int, int) = (64, 64),
        s1: int = 2,
        s2: int = 2,
    ) -> None:
        """Initialize patch expansion."""
        super().__init__()

        # Check size compatibilities
        try:
            msg = "New embedding dimension not divisible by patch-expansion factors!!!"
            assert (emb_size * emb_factor) % (s1 * s2) == 0, msg
        except AssertionError as e:
            msg_tuple = (
                "Input embedding size:",
                emb_size,
                "Embedding factor:",
                emb_factor,
                "Height expansion factor:",
                s1,
                "Width expansion factor:",
                s2,
            )
            e.args += msg_tuple
            raise

        # Input embedding size
        self.emb_size = emb_size

        # Patch grid parameters
        self.H = patch_grid_size[0]
        self.W = patch_grid_size[1]

        # Patch division factors
        self.s1 = s1
        self.s2 = s2

        # Embedding dimension factor
        self.emb_factor = emb_factor

        # New patch grid
        self.out_patch_grid_size = (int(self.H * self.s1), int(self.W * self.s2))

        # Add output embedding size for model building
        self.out_emb_size = int(self.emb_factor * self.emb_size / (self.s1 * self.s2))

        # Linear re-embedding
        self.linear = nn.Linear(self.emb_size, self.emb_factor * self.emb_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method for patch expansion."""
        # The input tensor is shape (B, num_tokens, embedding_dim)
        _, L, _ = x.shape

        # NOTE: The number of tokens is assumed to be L=H*W
        assert L == self.H * self.W

        # Linear embedding
        x = self.linear(x)

        # Rearrange
        x = rearrange(
            x,
            "b (h w) (k s1 s2) -> b (h s1 w s2) k",
            s1=self.s1,
            s2=self.s2,
            h=self.H,
            w=self.W,
        )

        return x


class Unpatchify(nn.Module):
    r"""Expansion from patches to variables and images.

    This layer performs the remap:

    .. math::

        (B, H \cdot W, V \cdot p_h \cdot p_w)
        \xrightarrow{\text{rearrange}}
        (B, V, H \cdot p_h, W \cdot p_w)

    Args:
        total_num_vars (int): Total number of variables to be output.
        patch_grid_size (int, int): Height and width grid size of patches
                                    making up the tokens.
        patch_size (int, int): Height and width of each patch.

    """

    def __init__(
        self,
        total_num_vars: int = 5,
        patch_grid_size: (int, int) = (64, 64),
        patch_size: (int, int) = (8, 8),
    ) -> None:
        """Initialization for Unpatchify."""
        super().__init__()

        # Total number of variables
        self.V = total_num_vars

        # Patch grid parameters
        self.H = patch_grid_size[0]
        self.W = patch_grid_size[1]

        # Individual patch height and width
        self.p_h = patch_size[0]
        self.p_w = patch_size[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method for Unpatchify."""
        # The input tensor is shape (B, num_tokens, embedding_dim)
        _, L, C = x.shape

        # Make sure shape requirements are met
        assert L == self.H * self.W

        assert C == self.V * self.p_h * self.p_w

        x = rearrange(
            x,
            "b (h w) (v ph pw) -> b v (h ph) (w pw)",
            h=self.H,
            w=self.W,
            v=self.V,
            ph=self.p_h,
            pw=self.p_w,
        )

        return x


if __name__ == "__main__":
    """Usage Example.

    """

    # Original input before embedding: (B, V, H, W)
    img_size = (512, 128)
    num_vars = 5
    batch_size = 3
    emb_dim = 64
    patch_size = (16, 8)
    patch_grid_size = (
        int(img_size[0] / patch_size[0]),
        int(img_size[1] / patch_size[1]),
    )
    num_tokens = patch_grid_size[0] * patch_grid_size[1]

    # Input
    x = torch.rand(batch_size, num_tokens, emb_dim)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = x.type(torch.FloatTensor).to(device)

    s1 = 4
    s2 = 4
    emb_factor = 2
    merge_model = PatchMerge(
        emb_dim, emb_factor=emb_factor, patch_grid_size=patch_grid_size, s1=s1, s2=s2
    ).to(device)

    print("Input shape:", x.shape)
    x = merge_model(x)
    print("Patch merge shape:", x.shape)

    # Grid size has been reduced through merge
    merged_patch_grid_size = (int(patch_grid_size[0] / s1), int(patch_grid_size[1] / s2))
    expand_model = PatchExpand(
        x.shape[2],
        emb_factor=int(s1 * s2 / emb_factor),
        patch_grid_size=merged_patch_grid_size,
        s1=s1,
        s2=s2,
    ).to(device)
    x = expand_model(x)
    print("Patch expand shape:", x.shape)

    # Linear embed the last dimension into V*p_h*p_w
    linear = nn.Linear(emb_dim, num_vars * patch_size[0] * patch_size[1]).to(device)
    x = linear(x)
    print("Embed to variable dimension shape:", x.shape)

    # Unpatch the variables and tokens
    unpatch_model = Unpatchify(
        total_num_vars=num_vars, patch_grid_size=patch_grid_size, patch_size=patch_size
    ).to(device)

    x = unpatch_model(x)
    print("Unpatched image and variables shape:", x.shape)
