"""Grouped-convolution embedding.

nn.Module classes defined here offer different methods to embed a multi-channel
image as a series of tokenized patches.

"""

import math

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange


def _get_conv2d_weights(
    in_channels: int, out_channels: int, kernel_size: tuple[int, int]
) -> torch.Tensor:
    """Set up tensor of proper dimensions for weights."""
    weight = torch.empty(out_channels, in_channels, *kernel_size)

    return weight


def _get_conv2d_biases(out_channels: int) -> torch.Tensor:
    """Set up tensor of proper dimensions for biases."""
    bias = torch.empty(out_channels)

    return bias


class ParallelVarPatchEmbed(nn.Module):
    """Parallel patch embedding.

    Variable to Patch Embedding with multiple variables in a single kernel. Key
    idea is to use Grouped Convolutions. This allows this layer to embed an
    arbitrary subset of a default list of variables.

    NOTE: The img_size entries should be divisible by the corresponding
    patch_size entries.

    Args:
        max_vars (int): Maximum number of variables
        img_size ((int, int)): Image size
        patch_size ((int, int)): Patch size
        embed_dim (int): Embedding dimension
        norm_layer (nn.Module, optional): Normalization layer. Defaults to None.

    """

    def __init__(
        self,
        max_vars: int = 5,
        img_size: (int, int) = (128, 128),
        patch_size: (int, int) = (16, 16),
        embed_dim: int = 64,
        norm_layer: Optional[nn.Module] = None,
    ) -> None:
        """Initialization for parallel embedding."""
        super().__init__()
        # Check size compatibilities
        try:
            msg = "Image height not divisible by patch height!!!"
            assert img_size[0] % patch_size[0] == 0, msg
        except AssertionError as e:
            msg_tuple = ("Image height:", img_size[0], "Patch height:", patch_size[0])
            e.args += msg_tuple
            raise

        try:
            msg = "Image width not divisible by patch width!!!"
            assert img_size[1] % patch_size[1] == 0, msg
        except AssertionError as e:
            msg_tuple = ("Image width:", img_size[1], "Patch width:", patch_size[1])
            e.args += msg_tuple
            raise

        self.max_vars = max_vars
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (
            img_size[0] // self.patch_size[0],
            img_size[1] // self.patch_size[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.embed_dim = embed_dim

        # Conv patch embedding weights and biases
        grouped_weights = torch.stack(
            [
                _get_conv2d_weights(1, self.embed_dim, self.patch_size)
                for _ in range(max_vars)
            ],
            dim=0,
        )

        self.proj_weights = nn.Parameter(grouped_weights)

        grouped_biases = torch.stack(
            [_get_conv2d_biases(self.embed_dim) for _ in range(max_vars)], dim=0
        )
        self.proj_biases = nn.Parameter(grouped_biases)

        # Layer normalization
        self.norm = norm_layer(self.embed_dim) if norm_layer else nn.Identity()

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights for projections."""
        for idx in range(self.max_vars):
            nn.init.kaiming_uniform_(self.proj_weights[idx], a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.proj_weights[idx])

            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.proj_biases[idx], -bound, bound)

    def forward(self, x: torch.Tensor, in_vars: torch.Tensor) -> torch.Tensor:
        """Forward method of the Parallel Embedding.

        NOTE: `in_vars` should be a (1, C) tensor of integers where the
        integers correspond to the variables present in the channels of
        `x`. This implies that every sample in the batch represented by `x`
        must correspond to the same variables.

        """
        weights = self.proj_weights[in_vars].flatten(0, 1)
        biases = self.proj_biases[in_vars].flatten(0, 1)

        # Each variable's image is embedded separately. Each variable's image
        # gets mapped to an a set of patches of embeddings. The variable's
        # embeddings are concatenated.
        #
        # For embedding dimension E and patch size p=(p1, p2),
        # (B, V, H, W) -> (B, VxE, H', W') with H'=H/p1, W'=W/p2
        groups = in_vars.shape[0]
        proj = F.conv2d(x, weights, biases, groups=groups, stride=self.patch_size)

        # Flatten the patch arrays and separate the variables and embeddings.
        proj = rearrange(
            proj, "b (v e) h1 h2 -> b v (h1 h2) e", v=groups, e=self.embed_dim
        )

        # Normalize the layer output
        proj = self.norm(proj)

        return proj


class SwinEmbedding(nn.Module):
    r"""SWIN patch embedding.

    This SWIN embedding layer takes a *channels-first* image, applies linear
    patch embeddings, and rearranges the resulting embedded patches into
    sequences of tokens.

    For an input tensor of shape :math:`(B, C, H, W)`, the embedding uses a
    convolutional kernel with embedding dimension :math:`E` and patch size
    :math:`P = (p_h, p_w)`. This operation produces an embedded tensor of shape
    :math:`(B, E, H', W')`, where:

    .. math::

        H' = \frac{H}{p_h}, \quad W' = \frac{W}{p_w}

    The resulting tensor is then rearranged into a sequence of
    :math:`H' \times W'` tokens, each of dimension :math:`E`, resulting in a
    final tensor of shape:

    .. math::

        (B, H' \cdot W', E)

    .. note::
        The values of :math:`H` and :math:`W` (i.e., `img_size`) must be
        divisible by their corresponding patch size dimensions :math:`p_h` and
        :math:`p_w`.

    Args:
        max_vars (int): Maximum number of variables
        img_size (Tuple[int, int]): Image size (height, width)
        patch_size (Tuple[int, int]): Patch size (height, width)
        embed_dim (int): Embedding dimension
        norm_layer (nn.Module, optional): Normalization layer. Defaults to None.

    """

    def __init__(
        self,
        num_vars: int = 5,
        img_size: (int, int) = (128, 128),
        patch_size: (int, int) = (16, 16),
        embed_dim: int = 64,
        norm_layer: Optional[nn.Module] = None,
    ) -> None:
        """Initialization for SWIN patch embedding."""
        super().__init__()
        # Check size compatibilities
        try:
            msg = "Image height not divisible by patch height!!!"
            assert img_size[0] % patch_size[0] == 0, msg
        except AssertionError as e:
            msg_tuple = ("Image height:", img_size[0], "Patch height:", patch_size[0])
            e.args += msg_tuple
            raise

        try:
            msg = "Image width not divisible by patch widht!!!"
            assert img_size[1] % patch_size[1] == 0, msg
        except AssertionError as e:
            msg_tuple = ("Image width:", img_size[1], "Patch width:", patch_size[1])
            e.args += msg_tuple
            raise

        self.num_vars = num_vars
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.grid_size = (
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.linear_embedding = nn.Conv2d(
            self.num_vars,
            self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.rearrange = Rearrange("b c h w -> b (h w) c")

        self.norm = norm_layer(self.embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method for SWIN patch embedding."""
        x = self.linear_embedding(x)
        x = self.rearrange(x)

        x = self.norm(x)

        return x


if __name__ == "__main__":
    """Usage Example.

    """

    # (B, C, H, W) = (3, 5, 128, 128)
    x = torch.rand(3, 4, 128, 128)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = x.type(torch.FloatTensor).to(device)
    in_vars = torch.tensor([0, 1, 3, 4]).to(device)
    PPembed_model = ParallelVarPatchEmbed(
        max_vars=5,
        img_size=(128, 128),
        patch_size=(16, 16),
        embed_dim=72,
        norm_layer=nn.LayerNorm,
    ).to(device)

    print("Input shape:", x.shape)
    print("Parallel-patch embed shape:", PPembed_model(x, in_vars=in_vars).shape)

    swin_embed_model = SwinEmbedding(
        num_vars=4,
        img_size=(128, 128),
        patch_size=(16, 16),
        embed_dim=72,
        norm_layer=nn.LayerNorm,
    ).to(device)

    print("Input shape:", x.shape)
    print("SWIN embed shape:", swin_embed_model(x).shape)
