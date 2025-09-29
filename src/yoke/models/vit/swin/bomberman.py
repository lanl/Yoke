"""Module for BomberMan network structures.

These architectures expand on the LodeRunner architecture to add multiple time inputs,
spatial-grid encoding, multiple time outputs, etc.

"""

from collections.abc import Iterable

import numpy as np
import torch
from torch import nn

from einops import rearrange

from yoke.models.vit.swin.unet import SwinUnetBackbone
from yoke.models.vit.patch_embed import ParallelVarPatchEmbed
from yoke.models.vit.patch_manipulation import Unpatchify
from yoke.models.vit.aggregate_variables import AggVars
from yoke.models.vit.embedding_encoders import (
    VarEmbed,
    PosEmbed,
    TimeEmbed,
)

from yoke.helpers.training_design import validate_patch_and_window


class BomberMan(nn.Module):
    """BomberMan neural network.

    Parallel-patch embedding with SWIN U-Net backbone and unpatchification. This module
    will take in multiple timesteps of variable-channel image format and output a single
    next timestep of an equivalent variable-channel image format. This represents a
    time-dependent advancement on the LodeRunner architecture.

    NOTE 1: The input size for a batch in LodeRunner is (B, C, H, W). Probably the most
    natural way to add time-dependency is to expect (B, T, C, H, W) as input.

    NOTE 2: LodeRunner encodes the time-step between the input and output images as an
    input. This class will assume every image in the input and output sequences are
    equally spaced in time.

    NOTE 3: This first BomberMan version will just map to a single next timestep.
    Subsequent iterations will allow multiple time-step outputs.

    Our approach will be as follows:

    Start with x: (B, T, C, H, W)
    Embed each frame with your existing patch-conv + channel-aggregate:
    x = rearrange(x, 'b t c h w -> (b t) c h w')
    x = your_patch_conv_and_channel_query(x)
    Now x is (B*T, L, d) so we map back to 4D: (B, T, L, d)
    x = rearrange(x, '(b t) L d -> b t L d')

    Now pass this through M-layers of factorized spatial-temporal MSA

    Spatial MSA over the L tokens in each frame:
    x_sp = rearrange(x, 'b t L d -> (b t) L d')
    x_sp = x_sp + time-SWINencoder2(LN(x_sp))        # (B*T, L, d)
    x = rearrange(x_sp, '(b t) L d -> b t L d')

    Temporal MSA over the T tokens at each spatial location
    x_tp = rearrange(x, 'b t L d -> (b L) t d')
    x_tp = x_tp + TemporalMSA(LN(x_tp))       # (B*L, T, d)
    x = rearrange(x_tp, '(b L) t d -> b t L d')

    # MLP after layer-normalization and residual connection as usual
    x = x + MLP(LN(x))
    Output is still (B, T, L, d)

    Collapse the time dimension to get a single output for each variable:
    with a learned query:
    x_read  = rearrange(x, 'b t L d -> (b L) t d')   # (B*L, T, d)
    q_time  = Parameter(torch.randn(1, 1, d))       # one learnable query
    q_time  = q_time.expand(b*L, -1, -1)            # (B*L, 1, d)
    y_read, _ = MultiheadAttention(d, heads)(q_time, x_read, x_read)
    y = rearrange(y_read, '(b L) 1 d -> b L d')     # (B, L, d)

    Args:
        default_vars (list[str]): List of default variables to be used for training
        image_size (tuple[int, int]): Height and width, in pixels, of input image.
        patch_size (tuple[int, int]): Height and width pixel dimensions of patch in
                                      initial embedding.
        emb_dim (int): Initial embedding dimension.
        emb_factor (int): Scale of embedding in each patch merge/expand.
        num_heads (int): Number of heads in the MSA layers.
        block_structure (int, int, int, int): Tuple specifying the number of SWIN
                                              encoders in each block structure
                                              separated by the patch-merge layers.
        window_sizes (list(4*(int, int))): Window sizes within each SWIN encoder/decoder.
        patch_merge_scales (list(3*(int, int))): Height and width scales used in
                                                 each patch-merge layer.
        verbose (bool): When TRUE, windowing and merging dimensions are printed
                        during initialization.

    """

    def __init__(
        self,
        default_vars: list[str],
        image_size: Iterable[int, int] = (1120, 800),
        patch_size: Iterable[int, int] = (10, 10),
        embed_dim: int = 128,
        # emb_factor: int = 2,
        # num_heads: int = 8,
        # block_structure: Iterable[int, int, int, int] = (1, 1, 3, 1),
        # window_sizes: Iterable[(int, int), (int, int), (int, int), (int, int)] = [
        #     (8, 8),
        #     (8, 8),
        #     (4, 4),
        #     (2, 2),
        # ],
        # patch_merge_scales: Iterable[(int, int), (int, int), (int, int)] = [
        #     (2, 2),
        #     (2, 2),
        #     (2, 2),
        # ],
        verbose: bool = False,
    ) -> None:
        """Initialization for class."""
        super().__init__()

        self.default_vars = default_vars
        self.max_vars = len(self.default_vars)
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        # self.emb_factor = emb_factor
        # self.num_heads = num_heads
        # self.block_structure = block_structure
        # self.window_sizes = window_sizes
        # self.patch_merge_scales = patch_merge_scales

        # # Validate patch_size, window_sizes, and patch_merge_scales before proceeding.
        # valid = validate_patch_and_window(
        #     image_size=image_size,
        #     patch_size=patch_size,
        #     window_sizes=window_sizes,
        #     patch_merge_scales=patch_merge_scales,
        # )
        # assert np.all(valid), (
        #     "Invalid combination of image_size, patch_size, window_sizes, "
        #     "and patch_merge_scales!"
        # )

        # First embed the image as a sequence of tokenized patches. Each
        # channel is embedded independently.
        self.parallel_embed = ParallelVarPatchEmbed(
            max_vars=self.max_vars,
            img_size=self.image_size,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            norm_layer=None,
        )

        # Encode tokens corresponding to each variable with a learnable tag
        self.var_embed_layer = VarEmbed(self.default_vars, self.embed_dim)

        # Aggregate variable tokenizations using an attention mechanism
        self.agg_vars = AggVars(self.embed_dim, self.num_heads)

        # Encode each patch with position information. Position encoding is
        # only index-aware and does not take into account actual spatial
        # information.
        self.pos_embed = PosEmbed(
            self.embed_dim,
            self.patch_size,
            self.image_size,
            self.parallel_embed.num_patches,
        )

        # # Pass encoded patch tokens through a SWIN-Unet structure
        # self.unet = SwinUnetBackbone(
        #     emb_size=self.embed_dim,
        #     emb_factor=self.emb_factor,
        #     patch_grid_size=self.parallel_embed.grid_size,
        #     block_structure=self.block_structure,
        #     num_heads=self.num_heads,
        #     window_sizes=self.window_sizes,
        #     patch_merge_scales=self.patch_merge_scales,
        #     verbose=verbose,
        # )

        # # Linear embed the last dimension into V*p_h*p_w
        # self.linear4unpatch = nn.Linear(
        #     self.embed_dim, self.max_vars * self.patch_size[0] * self.patch_size[1]
        # )

        # # Unmap the tokenized embeddings to variables and images.
        # self.unpatch = Unpatchify(
        #     total_num_vars=self.max_vars,
        #     patch_grid_size=self.parallel_embed.grid_size,
        #     patch_size=self.patch_size,
        # )

    def forward(
        self,
        x: torch.Tensor,
        in_vars: torch.Tensor,
        out_vars: torch.Tensor,
    ) -> torch.Tensor:
        """Forward method for BomberMan."""
        # WARNING!: Most likely the `in_vars` and `out_vars` need to be tensors
        # of integers corresponding to variables in the `default_vars` list.

        # x is expected to be of shape (B, T, C, H, W) where:
        #   B = batch size
        #   T = number of timesteps
        #   C = number of variables
        #   H = height of the image
        #   W = width of the image

        # Reshape input to (B*T, C, H, W) for parallel embedding
        B = x.shape[0]  # Batch size
        T = x.shape[1]  # Number of timesteps
        x = rearrange(x, 'b t c h w -> (b t) c h w')

        # First embed input
        x = self.parallel_embed(x, in_vars)  # (B*T, V, L=Hw*Ww, D)

        # Encode variables
        x = self.var_embed_layer(x, in_vars)  # (B*T, V, L, D)

        # Aggregate variables
        x = self.agg_vars(x)  # (B*T, L, D)

        # Encode patch positions, spatial information
        x = self.pos_embed(x)  # (B*T, L, D)

        # Reshape to (B, T, L, D) for temporal encoding
        x = rearrange(x, '(b t) L d -> b t L d', b=B, t=T)

        # # Here we insert the tSWIN (time-dependent SWIN) structure of factorized
        # # spatial-temporal SWIN-V2
        # x = self.tSWIN(x)  # (B, T, L, D)

        # # Aggregate the temporal dimension to get a single output using a learned query.
        # x = self.agg_temporal(x)  # (B, L, D)

        # # Pass through SWIN-V2 U-Net encoder
        # x = self.unet(x)

        # # Use linear map to remap to correct variable and patchsize dimension
        # x = self.linear4unpatch(x)

        # # Unpatchify back to original shape
        # x = self.unpatch(x)

        # # Select only entries corresponding to out_vars for loss
        # preds = x[:, out_vars]

        return x

