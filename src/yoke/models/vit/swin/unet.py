"""The SWIN U-Net backbone."""

import torch
from torch import nn

from yoke.models.vit.swin.encoder import SwinEncoder2, SwinConnectEncoder
from yoke.models.vit.swin.encoder import SwinConnectDecoder
from yoke.models.vit.patch_manipulation import PatchMerge, PatchExpand

from yoke.utils.parameters import count_torch_params


class SwinUnetBackbone(nn.Module):
    """SWIN U-Net architecture.

    This backbone has no initial patch embedding or terminal patch
    expansion. Instead this `nn.Module` just expects a (B, H*W, C) tensor
    resulting from some patch embedding. The output is then a (B, H*W, C)
    tensor.

    In this U-Net architecture the *left-arm* architecture mirrors that of the
    *right-arm*. Architetures for these arms can not be specified
    independently.

    Args:
        emb_size (int): Initial embedding dimension.
        emb_factor (int): Scale of embedding in each patch merge/exand.
        patch_grid_size (int, int): Initial patch-grid height and width for input.
        block_structure (int, int, int, int): Tuple specifying the number of SWIN
                                              encoders in each block structure
                                              separated by the patch-merge layers.
        num_heads (int): Number of heads in the MSA layers.
        window_sizes (list(4*(int, int))): Window sizes within each SWIN encoder/decoder.
        patch_merge_scales (list(3*(int, int))): Height and width scales used in
                                                 each patch-merge layer.
        verbose (bool): When TRUE, windowing and merging dimensions are printed
                        during initialization.

    """

    def __init__(
        self,
        emb_size: int = 96,
        emb_factor: int = 2,
        patch_grid_size: (int, int) = (112, 80),
        block_structure: (int, int, int, int) = (1, 1, 3, 1),
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
        """Initialization for SWIN-UNet."""
        super().__init__()
        # Assign inputs as attributes of transformer
        self.emb_size = emb_size
        self.emb_factor = emb_factor
        self.patch_grid_size = patch_grid_size
        self.block_structure = block_structure
        self.num_heads = num_heads

        self.window_sizes = window_sizes
        self.patch_merge_scales = patch_merge_scales
        self.num_output_classes = num_output_classes

        # Set up lists of encoding/decoding, merge/expand layers
        self.dwn_stage1 = nn.ModuleList()
        self.dwn_stage2 = nn.ModuleList()
        self.dwn_stage3 = nn.ModuleList()

        self.bottleneck_stage4 = nn.ModuleList()

        self.up_stage1 = nn.ModuleList()
        self.up_stage2 = nn.ModuleList()
        self.up_stage3 = nn.ModuleList()

        self.down_connect = nn.ModuleList()
        self.up_connect = nn.ModuleList()
        self.PatchMerge = nn.ModuleList()
        self.PatchExpand = nn.ModuleList()

        # Lists to track DOWN-arm size changes
        dwn_emb_size_list = []
        dwn_num_heads_list = []
        dwn_patch_grid_size_list = []
        dwn_window_size_list = []
        dwn_patch_merge_scale_list = []
        block_structure_list = []

        ############################
        # DOWN
        ############################
        # Add sizes to lists
        dwn_emb_size_list.append(self.emb_size)
        dwn_num_heads_list.append(self.num_heads)
        dwn_patch_grid_size_list.append(self.patch_grid_size)
        dwn_window_size_list.append(self.window_sizes[0])
        dwn_patch_merge_scale_list.append(self.patch_merge_scales[0])
        block_structure_list.append(self.block_structure[0])

        # A series of SWIN encoders with embedding size and number of heads
        # doubled every stage.
        for i in range(self.block_structure[0] - 1):
            self.dwn_stage1.append(
                SwinEncoder2(
                    emb_size=self.emb_size,
                    num_heads=self.num_heads,
                    patch_grid_size=self.patch_grid_size,
                    window_size=self.window_sizes[0],
                )
            )

            # Add additional layer normalization every 3 encoder blocks
            if (i + 1) % 3 == 0:
                self.dwn_stage1.append(nn.LayerNorm(self.emb_size))

        # Last entry of block adds skip connection
        self.down_connect.append(
            SwinConnectEncoder(
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

        # Add sizes to lists
        dwn_emb_size_list.append(new_emb_size)
        dwn_num_heads_list.append(new_num_heads)
        dwn_patch_grid_size_list.append(new_patch_grid_size)
        dwn_window_size_list.append(self.window_sizes[1])
        dwn_patch_merge_scale_list.append(self.patch_merge_scales[1])
        block_structure_list.append(self.block_structure[1])

        for i in range(self.block_structure[1] - 1):
            self.dwn_stage2.append(
                SwinEncoder2(
                    emb_size=new_emb_size,
                    num_heads=new_num_heads,
                    patch_grid_size=new_patch_grid_size,
                    window_size=self.window_sizes[1],
                )
            )

            # Add additional layer normalization every 3 encoder blocks
            if (i + 1) % 3 == 0:
                self.dwn_stage2.append(nn.LayerNorm(new_emb_size))

        # Last entry of block adds skip connection
        self.down_connect.append(
            SwinConnectEncoder(
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

        # Add sizes to lists
        dwn_emb_size_list.append(new_emb_size)
        dwn_num_heads_list.append(new_num_heads)
        dwn_patch_grid_size_list.append(new_patch_grid_size)
        dwn_window_size_list.append(self.window_sizes[2])
        dwn_patch_merge_scale_list.append(self.patch_merge_scales[2])
        block_structure_list.append(self.block_structure[2])

        for i in range(self.block_structure[2] - 1):
            self.dwn_stage3.append(
                SwinEncoder2(
                    emb_size=new_emb_size,
                    num_heads=new_num_heads,
                    patch_grid_size=new_patch_grid_size,
                    window_size=self.window_sizes[2],
                )
            )

            # Add additional layer normalization every 3 encoder blocks
            if (i + 1) % 3 == 0:
                self.dwn_stage3.append(nn.LayerNorm(new_emb_size))

        # Last entry of block adds skip connection
        self.down_connect.append(
            SwinConnectEncoder(
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

        ############################
        # BOTTLENECK
        ############################
        new_patch_grid_size = self.PatchMerge[-1].out_patch_grid_size
        new_emb_size = self.PatchMerge[-1].out_emb_size
        new_num_heads = self.emb_factor * new_num_heads
        if verbose:
            print("New patch-grid size after merge 3:", new_patch_grid_size)
            print("New embedding size after merge 3:", new_emb_size)
            print("New number of heads after merge 3:", new_num_heads)

        # Add sizes to lists
        dwn_emb_size_list.append(new_emb_size)
        dwn_num_heads_list.append(new_num_heads)
        dwn_patch_grid_size_list.append(new_patch_grid_size)
        dwn_window_size_list.append(self.window_sizes[3])

        for i in range(self.block_structure[3]):
            self.bottleneck_stage4.append(
                SwinEncoder2(
                    emb_size=new_emb_size,
                    num_heads=new_num_heads,
                    patch_grid_size=new_patch_grid_size,
                    window_size=self.window_sizes[3],
                )
            )

            # Add additional layer normalization every 3 encoder blocks
            if (i + 1) % 3 == 0:
                self.bottleneck_stage4.append(nn.LayerNorm(new_emb_size))

        ############################
        # UP
        ############################
        # Stage 1
        # Expand patches, undoing original patch merge
        emb_size = dwn_emb_size_list.pop()
        num_heads = dwn_num_heads_list.pop()
        patch_grid_size = dwn_patch_grid_size_list.pop()
        window_size = dwn_window_size_list.pop()
        merge_scale = dwn_patch_merge_scale_list.pop()
        s1 = merge_scale[0]
        s2 = merge_scale[1]
        self.PatchExpand.append(
            PatchExpand(
                emb_size=emb_size,
                emb_factor=int(s1 * s2 / self.emb_factor),
                patch_grid_size=patch_grid_size,
                s1=s1,
                s2=s2,
            )
        )

        # Get new sizes
        new_emb_size = self.PatchExpand[-1].out_emb_size
        new_patch_grid_size = self.PatchExpand[-1].out_patch_grid_size
        # Skip connection receptor
        self.up_connect.append(
            SwinConnectDecoder(
                emb_size=new_emb_size,
                num_heads=num_heads,
                patch_grid_size=new_patch_grid_size,
                window_size=window_size,
            )
        )

        # Reverse the DOWN-arm encoder process
        for i in range(block_structure_list.pop() - 1):
            self.up_stage1.append(
                SwinEncoder2(
                    emb_size=new_emb_size,
                    num_heads=num_heads,
                    patch_grid_size=new_patch_grid_size,
                    window_size=window_size,
                )
            )

            # Add additional layer normalization every 3 encoder blocks
            if (i + 1) % 3 == 0:
                self.up_stage1.append(nn.LayerNorm(new_emb_size))

        # Stage 2
        # Expand patches, undoing original patch merge
        emb_size = dwn_emb_size_list.pop()
        num_heads = dwn_num_heads_list.pop()
        patch_grid_size = dwn_patch_grid_size_list.pop()
        window_size = dwn_window_size_list.pop()
        merge_scale = dwn_patch_merge_scale_list.pop()
        s1 = merge_scale[0]
        s2 = merge_scale[1]
        self.PatchExpand.append(
            PatchExpand(
                emb_size=emb_size,
                emb_factor=int(s1 * s2 / self.emb_factor),
                patch_grid_size=patch_grid_size,
                s1=s1,
                s2=s2,
            )
        )

        # Get new sizes
        new_emb_size = self.PatchExpand[-1].out_emb_size
        new_patch_grid_size = self.PatchExpand[-1].out_patch_grid_size
        # Skip connection receptor
        self.up_connect.append(
            SwinConnectDecoder(
                emb_size=new_emb_size,
                num_heads=num_heads,
                patch_grid_size=new_patch_grid_size,
                window_size=window_size,
            )
        )

        # Reverse the DOWN-arm encoder process
        for i in range(block_structure_list.pop() - 1):
            self.up_stage2.append(
                SwinEncoder2(
                    emb_size=new_emb_size,
                    num_heads=num_heads,
                    patch_grid_size=new_patch_grid_size,
                    window_size=window_size,
                )
            )

            # Add additional layer normalization every 3 encoder blocks
            if (i + 1) % 3 == 0:
                self.up_stage2.append(nn.LayerNorm(new_emb_size))

        # Stage 3
        # Expand patches, undoing original patch merge
        emb_size = dwn_emb_size_list.pop()
        num_heads = dwn_num_heads_list.pop()
        patch_grid_size = dwn_patch_grid_size_list.pop()
        window_size = dwn_window_size_list.pop()
        merge_scale = dwn_patch_merge_scale_list.pop()
        s1 = merge_scale[0]
        s2 = merge_scale[1]
        self.PatchExpand.append(
            PatchExpand(
                emb_size=emb_size,
                emb_factor=int(s1 * s2 / self.emb_factor),
                patch_grid_size=patch_grid_size,
                s1=s1,
                s2=s2,
            )
        )

        # Get new sizes
        new_emb_size = self.PatchExpand[-1].out_emb_size
        new_patch_grid_size = self.PatchExpand[-1].out_patch_grid_size
        # Skip connection receptor
        self.up_connect.append(
            SwinConnectDecoder(
                emb_size=new_emb_size,
                num_heads=num_heads,
                patch_grid_size=new_patch_grid_size,
                window_size=window_size,
            )
        )

        # Reverse the DOWN-arm encoder process
        for i in range(block_structure_list.pop() - 1):
            self.up_stage3.append(
                SwinEncoder2(
                    emb_size=new_emb_size,
                    num_heads=num_heads,
                    patch_grid_size=new_patch_grid_size,
                    window_size=window_size,
                )
            )

            # Add additional layer normalization every 3 encoder blocks
            if (i + 1) % 3 == 0:
                self.up_stage3.append(nn.LayerNorm(new_emb_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method for SWIN-UNet."""
        # DOWN
        # Set up list of down-sample skip connections
        x_downsample = []

        # Stage 1 down
        # Enumeration of nn.moduleList is supported under `torch.jit.script`
        for i, stage in enumerate(self.dwn_stage1):
            x = stage(x)

        x, y = self.down_connect[0](x)
        x_downsample.append(y)
        x = self.PatchMerge[0](x)

        # Stage 2 down
        for i, stage in enumerate(self.dwn_stage2):
            x = stage(x)

        x, y = self.down_connect[1](x)
        x_downsample.append(y)
        x = self.PatchMerge[1](x)

        # Stage 3 down
        for i, stage in enumerate(self.dwn_stage3):
            x = stage(x)

        x, y = self.down_connect[2](x)
        x_downsample.append(y)
        x = self.PatchMerge[2](x)

        # BOTTLENECK
        for i, stage in enumerate(self.bottleneck_stage4):
            x = stage(x)

        # UP
        # Stage 1 up
        x = self.PatchExpand[0](x)
        x = self.up_connect[0](x, x_downsample.pop())

        for i, stage in enumerate(self.up_stage1):
            x = stage(x)

        # Stage 2 up
        x = self.PatchExpand[1](x)
        x = self.up_connect[1](x, x_downsample.pop())

        for i, stage in enumerate(self.up_stage2):
            x = stage(x)

        # Stage 3 up
        x = self.PatchExpand[2](x)
        x = self.up_connect[2](x, x_downsample.pop())

        for i, stage in enumerate(self.up_stage3):
            x = stage(x)

        return x


if __name__ == "__main__":
    # (B, H*W, C)
    x = torch.rand(5, 112 * 80, 96)  # 112*80=8960

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = x.type(torch.FloatTensor).to(device)

    # Swin U-net
    swin_t_unet = SwinUnetBackbone(
        emb_size=96,
        emb_factor=2,
        patch_grid_size=(112, 80),
        block_structure=(1, 1, 3, 1),
        num_heads=8,
        window_sizes=[(8, 8), (8, 8), (4, 4), (2, 2)],
        patch_merge_scales=[(2, 2), (2, 2), (2, 2)],
        num_output_classes=10,
        verbose=True,
    ).to(device)
    print("SWIN-T U-Net input shape:", x.shape)
    print("SWIN-T U-Net output shape:", swin_t_unet(x).shape)
    print("SWIN-T parameters:", count_torch_params(swin_t_unet, trainable=True))
