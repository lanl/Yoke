"""The SWIN transformer backbone."""

import torch
from torch import nn

from yoke.models.vit.swin.encoder import SwinEncoder, SwinEncoder2
from yoke.models.vit.patch_embed import SwinEmbedding
from yoke.models.vit.patch_manipulation import PatchMerge

from yoke.utils.parameters import count_torch_params


class Swin(nn.Module):
    """Main SWIN architecture from the original 2021 SWIN paper.

    This takes in an image and predicts a vector of size `num_output_classes`.

    Args:
        input_channels (int): Number of variables/channels input
        img_size (int, int): Input image height and width
        patch_size (int, int): Original patch-embedding patch size.
        block_structure (int, int, int, int): Tuple specifying the number of SWIN
                                              encoders in each block structure
                                              separated by the patch-merge layers.
        emb_size (int): Initial embedding dimension.
        emb_factor (int): Scale of embedding dimension in each patch merging.
        num_heads (int): Number of heads in the MSA layers.
        window_sizes (list(4*(int, int))): Window sizes within each SWIN encoder.
        patch_merge_scales (list(3*(int, int))): Height and width scales used in
                                                 each patch-merge layer.
        num_output_classes (int): Output dimension of SWIN
        verbose (bool): When TRUE, windowing and merging dimensions are printed
                        during initialization.

    """

    def __init__(
        self,
        input_channels: int = 3,
        img_size: (int, int) = (1120, 800),
        patch_size: (int, int) = (10, 10),
        block_structure: (int, int, int, int) = (1, 1, 3, 1),
        emb_size: int = 96,
        emb_factor: int = 2,
        num_heads: int = 10,
        window_sizes: [(int, int), (int, int), (int, int), (int, int)] = [
            (8, 8),
            (8, 8),
            (4, 4),
            (2, 2),
        ],
        patch_merge_scales: [(int, int), (int, int), (int, int)] = [
            (2, 2),
            (2, 2),
            (2, 2),
        ],
        num_output_classes: int = 5,
        verbose: bool = False,
    ) -> None:
        """Initialization for SWIN."""
        super().__init__()
        # Assign inputs as attributes of transformer
        self.input_channels = input_channels
        self.img_size = img_size
        self.patch_size = patch_size
        self.block_structure = block_structure
        self.emb_size = emb_size
        self.emb_factor = emb_factor
        self.num_heads = num_heads

        self.window_sizes = window_sizes
        self.patch_merge_scales = patch_merge_scales
        self.num_output_classes = num_output_classes

        # Embedding takes channels-first image (B, C, H, W) and returns patch
        # tokens (B, H'*W', E) with H'=H/ph, W'=W/pw for patch_size=(ph, pw).
        self.Embedding = SwinEmbedding(
            num_vars=self.input_channels,
            img_size=self.img_size,
            patch_size=self.patch_size,
            embed_dim=self.emb_size,
            norm_layer=None,
        )

        # Need to know the patch-grid dimensions a prior since we can have
        # rectangular images and patch grids.
        self.patch_grid_size = self.Embedding.grid_size

        if verbose:
            print("Input image size:", self.img_size)
            print("Patch-grid size after embedding:", self.patch_grid_size)

        # Set up list of encoding and merging layers
        self.stage1 = nn.ModuleList()
        self.stage2 = nn.ModuleList()
        self.stage3 = nn.ModuleList()
        self.stage4 = nn.ModuleList()

        self.PatchMerge = nn.ModuleList()

        # A series of SWIN encoders with embedding size and number of heads
        # doubled every stage.
        for i in range(self.block_structure[0]):
            self.stage1.append(
                SwinEncoder(
                    emb_size=self.emb_size,
                    num_heads=self.num_heads,
                    patch_grid_size=self.patch_grid_size,
                    window_size=self.window_sizes[0],
                )
            )

        # Patch-merging between each SWIN encoder block.
        self.PatchMerge.append(
            PatchMerge(
                emb_size=self.emb_size,
                emb_factor=self.emb_factor,
                patch_grid_size=self.patch_grid_size,
                s1=self.patch_merge_scales[0][0],
                s2=self.patch_merge_scales[0][1],
            )
        )

        new_patch_grid_size = self.PatchMerge[-1].out_patch_grid_size
        new_emb_size = self.PatchMerge[-1].out_emb_size
        new_num_heads = self.emb_factor * self.num_heads
        if verbose:
            print("New patch-grid size after merge 1:", new_patch_grid_size)
            print("New embedding size after merge 1:", new_emb_size)
            print("New number of heads after merge 1:", new_num_heads)

        for i in range(self.block_structure[1]):
            self.stage2.append(
                SwinEncoder(
                    emb_size=new_emb_size,
                    num_heads=new_num_heads,
                    patch_grid_size=new_patch_grid_size,
                    window_size=self.window_sizes[1],
                )
            )

        self.PatchMerge.append(
            PatchMerge(
                emb_size=new_emb_size,
                emb_factor=self.emb_factor,
                patch_grid_size=new_patch_grid_size,
                s1=self.patch_merge_scales[1][0],
                s2=self.patch_merge_scales[1][1],
            )
        )

        new_patch_grid_size = self.PatchMerge[-1].out_patch_grid_size
        new_emb_size = self.PatchMerge[-1].out_emb_size
        new_num_heads = self.emb_factor * new_num_heads
        if verbose:
            print("New patch-grid size after merge 2:", new_patch_grid_size)
            print("New embedding size after merge 2:", new_emb_size)
            print("New number of heads after merge 2:", new_num_heads)

        for i in range(self.block_structure[2]):
            self.stage3.append(
                SwinEncoder(
                    emb_size=new_emb_size,
                    num_heads=new_num_heads,
                    patch_grid_size=new_patch_grid_size,
                    window_size=self.window_sizes[2],
                )
            )

        self.PatchMerge.append(
            PatchMerge(
                emb_size=new_emb_size,
                emb_factor=self.emb_factor,
                patch_grid_size=new_patch_grid_size,
                s1=self.patch_merge_scales[2][0],
                s2=self.patch_merge_scales[2][1],
            )
        )

        new_patch_grid_size = self.PatchMerge[-1].out_patch_grid_size
        new_emb_size = self.PatchMerge[-1].out_emb_size
        new_num_heads = self.emb_factor * new_num_heads
        if verbose:
            print("New patch-grid size after merge 3:", new_patch_grid_size)
            print("New embedding size after merge 3:", new_emb_size)
            print("New number of heads after merge 3:", new_num_heads)

        for i in range(self.block_structure[3]):
            self.stage4.append(
                SwinEncoder(
                    emb_size=new_emb_size,
                    num_heads=new_num_heads,
                    patch_grid_size=new_patch_grid_size,
                    window_size=self.window_sizes[3],
                )
            )

        # All tokens are pooled using Adaptive Pooling
        self.avgpool1d = nn.AdaptiveAvgPool1d(output_size=1)

        self.layer = nn.Linear(new_emb_size, num_output_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method for SWIN."""
        x = self.Embedding(x)

        # enumeration of nn.moduleList is supported under `torch.jit.script`
        for i, stage in enumerate(self.stage1):
            x = stage(x)

        x = self.PatchMerge[0](x)

        for i, stage in enumerate(self.stage2):
            x = stage(x)

        x = self.PatchMerge[1](x)

        for i, stage in enumerate(self.stage3):
            x = stage(x)

        x = self.PatchMerge[2](x)

        for i, stage in enumerate(self.stage4):
            x = stage(x)

        x = self.layer(self.avgpool1d(x.transpose(1, 2)).squeeze(2))

        return x


class SwinV2(nn.Module):
    """Main SWIN-V2 architecture.

    This adds a learnable per-head scaling cosine attention in the MSA layers
    and swaps the order of the layer-normalization and the MSA/MLP layers in
    the encoders. Within each block of SWIN encoders, an extra layer
    normalization is added after every 3 encoders. These changes are reported
    to allow stable training of billion-parameter models.

    Args:
        input_channels (int): Number of variables/channels input
        img_size (int, int): Input image height and width
        patch_size (int, int): Original patch-embedding patch size.
        block_structure (int, int, int, int): Tuple specifying the number of SWIN
                                              encoders in each block structure
                                              separated by the patch-merge layers.
        emb_size (int): Initial embedding dimension.
        emb_factor (int): Scale of embedding dimension in each patch merging.
        num_heads (int): Number of heads in the MSA layers.
        window_sizes (list(4*(int, int))): Window sizes within each SWIN encoder.
        patch_merge_scales (list(3*(int, int))): Height and width scales used in
                                                 each patch-merge layer.
        num_output_classes (int): Output dimension of SWIN
        verbose (bool): When TRUE, windowing and merging dimensions are printed
                        during initialization.

    """

    def __init__(
        self,
        input_channels: int = 3,
        img_size: (int, int) = (1120, 800),
        patch_size: (int, int) = (10, 10),
        block_structure: (int, int, int, int) = (1, 1, 3, 1),
        emb_size: int = 96,
        emb_factor: int = 2,
        num_heads: int = 10,
        window_sizes: [(int, int), (int, int), (int, int), (int, int)] = [
            (8, 8),
            (8, 8),
            (4, 4),
            (2, 2),
        ],
        patch_merge_scales: [(int, int), (int, int), (int, int)] = [
            (2, 2),
            (2, 2),
            (2, 2),
        ],
        num_output_classes: int = 5,
        verbose: bool = False,
    ) -> None:
        """Initialization for SWIN-V2."""
        super().__init__()
        # Assign inputs as attributes of transformer
        self.input_channels = input_channels
        self.img_size = img_size
        self.patch_size = patch_size
        self.block_structure = block_structure
        self.emb_size = emb_size
        self.emb_factor = emb_factor
        self.num_heads = num_heads

        self.window_sizes = window_sizes
        self.patch_merge_scales = patch_merge_scales
        self.num_output_classes = num_output_classes

        # Embedding takes channels-first image (B, C, H, W) and returns patch
        # tokens (B, H'*W', E) with H'=H/ph, W'=W/pw for patch_size=(ph, pw).
        self.Embedding = SwinEmbedding(
            num_vars=self.input_channels,
            img_size=self.img_size,
            patch_size=self.patch_size,
            embed_dim=self.emb_size,
            norm_layer=None,
        )

        # Need to know the patch-grid dimensions a prior since we can have
        # rectangular images and patch grids.
        self.patch_grid_size = self.Embedding.grid_size

        if verbose:
            print("Input image size:", self.img_size)
            print("Patch-grid size after embedding:", self.patch_grid_size)

        # Set up list of encoding and merging layers
        self.stage1 = nn.ModuleList()
        self.stage2 = nn.ModuleList()
        self.stage3 = nn.ModuleList()
        self.stage4 = nn.ModuleList()

        self.PatchMerge = nn.ModuleList()

        # A series of SWIN encoders with embedding size and number of heads
        # doubled every stage.
        for i in range(self.block_structure[0]):
            self.stage1.append(
                SwinEncoder2(
                    emb_size=self.emb_size,
                    num_heads=self.num_heads,
                    patch_grid_size=self.patch_grid_size,
                    window_size=self.window_sizes[0],
                )
            )

            # Add additional layer normalization every 3 encoder blocks
            if (i + 1) % 3 == 0:
                self.stage1.append(nn.LayerNorm(self.emb_size))

        # Patch-merging between each SWIN encoder block.
        self.PatchMerge.append(
            PatchMerge(
                emb_size=self.emb_size,
                emb_factor=self.emb_factor,
                patch_grid_size=self.patch_grid_size,
                s1=self.patch_merge_scales[0][0],
                s2=self.patch_merge_scales[0][1],
            )
        )

        new_patch_grid_size = self.PatchMerge[-1].out_patch_grid_size
        new_emb_size = self.PatchMerge[-1].out_emb_size
        new_num_heads = self.emb_factor * self.num_heads
        if verbose:
            print("New patch-grid size after merge 1:", new_patch_grid_size)
            print("New embedding size after merge 1:", new_emb_size)
            print("New number of heads after merge 1:", new_num_heads)

        for i in range(self.block_structure[1]):
            self.stage2.append(
                SwinEncoder2(
                    emb_size=new_emb_size,
                    num_heads=new_num_heads,
                    patch_grid_size=new_patch_grid_size,
                    window_size=self.window_sizes[1],
                )
            )

            # Add additional layer normalization every 3 encoder blocks
            if (i + 1) % 3 == 0:
                self.stage2.append(nn.LayerNorm(new_emb_size))

        self.PatchMerge.append(
            PatchMerge(
                emb_size=new_emb_size,
                emb_factor=self.emb_factor,
                patch_grid_size=new_patch_grid_size,
                s1=self.patch_merge_scales[1][0],
                s2=self.patch_merge_scales[1][1],
            )
        )

        new_patch_grid_size = self.PatchMerge[-1].out_patch_grid_size
        new_emb_size = self.PatchMerge[-1].out_emb_size
        new_num_heads = self.emb_factor * new_num_heads
        if verbose:
            print("New patch-grid size after merge 2:", new_patch_grid_size)
            print("New embedding size after merge 2:", new_emb_size)
            print("New number of heads after merge 2:", new_num_heads)

        for i in range(self.block_structure[2]):
            self.stage3.append(
                SwinEncoder2(
                    emb_size=new_emb_size,
                    num_heads=new_num_heads,
                    patch_grid_size=new_patch_grid_size,
                    window_size=self.window_sizes[2],
                )
            )
            # Add additional layer normalization every 3 encoder blocks
            if (i + 1) % 3 == 0:
                self.stage3.append(nn.LayerNorm(new_emb_size))

        self.PatchMerge.append(
            PatchMerge(
                emb_size=new_emb_size,
                emb_factor=self.emb_factor,
                patch_grid_size=new_patch_grid_size,
                s1=self.patch_merge_scales[2][0],
                s2=self.patch_merge_scales[2][1],
            )
        )

        new_patch_grid_size = self.PatchMerge[-1].out_patch_grid_size
        new_emb_size = self.PatchMerge[-1].out_emb_size
        new_num_heads = self.emb_factor * new_num_heads
        if verbose:
            print("New patch-grid size after merge 3:", new_patch_grid_size)
            print("New embedding size after merge 3:", new_emb_size)
            print("New number of heads after merge 3:", new_num_heads)

        for i in range(self.block_structure[3]):
            self.stage4.append(
                SwinEncoder2(
                    emb_size=new_emb_size,
                    num_heads=new_num_heads,
                    patch_grid_size=new_patch_grid_size,
                    window_size=self.window_sizes[3],
                )
            )

            # Add additional layer normalization every 3 encoder blocks
            if (i + 1) % 3 == 0:
                self.stage4.append(nn.LayerNorm(new_emb_size))

        # All tokens are pooled using Adaptive Pooling
        self.avgpool1d = nn.AdaptiveAvgPool1d(output_size=1)

        self.layer = nn.Linear(new_emb_size, num_output_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method for SWIN-V2."""
        x = self.Embedding(x)

        # enumeration of nn.moduleList is supported under `torch.jit.script`
        for i, stage in enumerate(self.stage1):
            x = stage(x)

        x = self.PatchMerge[0](x)

        for i, stage in enumerate(self.stage2):
            x = stage(x)

        x = self.PatchMerge[1](x)

        for i, stage in enumerate(self.stage3):
            x = stage(x)

        x = self.PatchMerge[2](x)

        for i, stage in enumerate(self.stage4):
            x = stage(x)

        x = self.layer(self.avgpool1d(x.transpose(1, 2)).squeeze(2))

        return x


if __name__ == "__main__":
    # (B, C, H, W)
    x = torch.rand(5, 25, 1120, 800)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = x.type(torch.FloatTensor).to(device)

    # Model architectures come from the original SWIN papers by Liu et al.
    swin_t = Swin(
        input_channels=x.shape[1],
        img_size=(x.shape[2], x.shape[3]),
        patch_size=(10, 10),
        block_structure=(1, 1, 3, 1),
        emb_size=96,
        emb_factor=2,
        num_heads=8,
        window_sizes=[(8, 8), (8, 8), (4, 4), (2, 2)],
        patch_merge_scales=[(2, 2), (2, 2), (2, 2)],
        num_output_classes=10,
        verbose=True,
    ).to(device)
    print("SWIN-T output shape:", swin_t(x).shape)
    print("SWIN-T parameters:", count_torch_params(swin_t, trainable=True))

    swin_s = Swin(
        input_channels=x.shape[1],
        img_size=(x.shape[2], x.shape[3]),
        patch_size=(10, 10),
        block_structure=(1, 1, 9, 1),
        emb_size=96,
        emb_factor=2,
        num_heads=8,
        window_sizes=[(8, 8), (8, 8), (4, 4), (2, 2)],
        patch_merge_scales=[(2, 2), (2, 2), (2, 2)],
        num_output_classes=10,
    ).to(device)
    print("SWIN-S parameters:", count_torch_params(swin_s, trainable=True))

    swin_b = Swin(
        input_channels=x.shape[1],
        img_size=(x.shape[2], x.shape[3]),
        patch_size=(10, 10),
        block_structure=(1, 1, 9, 1),
        emb_size=128,
        emb_factor=2,
        num_heads=8,
        window_sizes=[(8, 8), (8, 8), (4, 4), (2, 2)],
        patch_merge_scales=[(2, 2), (2, 2), (2, 2)],
        num_output_classes=10,
    ).to(device)
    print("SWIN-B parameters:", count_torch_params(swin_b, trainable=True))

    swin_l = Swin(
        input_channels=x.shape[1],
        img_size=(x.shape[2], x.shape[3]),
        patch_size=(10, 10),
        block_structure=(1, 1, 9, 1),
        emb_size=192,
        emb_factor=2,
        num_heads=8,
        window_sizes=[(8, 8), (8, 8), (4, 4), (2, 2)],
        patch_merge_scales=[(2, 2), (2, 2), (2, 2)],
        num_output_classes=10,
    ).to(device)
    print("SWIN-L parameters:", count_torch_params(swin_l, trainable=True))

    swin_h = Swin(
        input_channels=x.shape[1],
        img_size=(x.shape[2], x.shape[3]),
        patch_size=(10, 10),
        block_structure=(1, 1, 9, 1),
        emb_size=352,
        emb_factor=2,
        num_heads=8,
        window_sizes=[(8, 8), (8, 8), (4, 4), (2, 2)],
        patch_merge_scales=[(2, 2), (2, 2), (2, 2)],
        num_output_classes=10,
    ).to(device)
    print("SWIN-H parameters:", count_torch_params(swin_h, trainable=True))

    swin_g = Swin(
        input_channels=x.shape[1],
        img_size=(x.shape[2], x.shape[3]),
        patch_size=(10, 10),
        block_structure=(1, 1, 11, 2),
        emb_size=512,
        emb_factor=2,
        num_heads=8,
        window_sizes=[(8, 8), (8, 8), (4, 4), (2, 2)],
        patch_merge_scales=[(2, 2), (2, 2), (2, 2)],
        num_output_classes=10,
    ).to(device)
    print("SWIN-G parameters:", count_torch_params(swin_g, trainable=True))

    swinV2_g = SwinV2(
        input_channels=x.shape[1],
        img_size=(x.shape[2], x.shape[3]),
        patch_size=(10, 10),
        block_structure=(1, 1, 11, 2),
        emb_size=512,
        emb_factor=2,
        num_heads=8,
        window_sizes=[(8, 8), (8, 8), (4, 4), (2, 2)],
        patch_merge_scales=[(2, 2), (2, 2), (2, 2)],
        num_output_classes=10,
    ).to(device)
    print("SWIN-V2-G parameters:", count_torch_params(swinV2_g, trainable=True))
