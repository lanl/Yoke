"""The main encoder structure used in a SWIN transformer."""

import torch
from torch import nn

from yoke.models.vit.swin.windowed_msa import WindowMSA, ShiftedWindowMSA
from yoke.models.vit.swin.windowed_msa import WindowCosMSA, ShiftedWindowCosMSA


class MLP(nn.Module):
    """A standard multi-layer perceptron structure using a GELU activtion.

    Consists of one hidden layer, and expanding the embedding size by 4x before
    contracting again.

    This MLP structure is specific to reproduce the SWIN-V2 architecture.

    Args:
        emb_size (int): Embedding layer dimension from input layer.

    """

    def __init__(self, emb_size: int = 64) -> None:
        """Initialization for MLP."""
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(emb_size, 4 * emb_size),
            nn.GELU(),
            nn.Linear(4 * emb_size, emb_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method for MLP."""
        return self.ff(x)


class SwinEncoder(nn.Module):
    """The main SWIN encoder.

    The main SWIN encoder alternates between a windowed-MSA and
    shifted-windowed-MSA block. MLP layers are used between and layer
    normalization is used prior to each layer. Residual connections are also
    included at each layer.

    Embedding size is the input dimension of the tokens. The embedding size
    must be evenly divisible by the number of heads. Moreover, the number of
    tokens, L, must satisfy L=patch_grid_size[0]*patch_grid_size[1]. The
    respective `window_size` dimensions must divide the `patch_grid_size`
    dimensions evenly.

    Args:
        emb_size (int): Incoming embedding dimension.
        num_heads (int): Number of heads to use in the MSA.
        patch_grid_size (int, int): Grid dimensions making up the token list.
        window_size (int, int): Dimensions of window to use on the patch grid.

    """

    def __init__(
        self,
        emb_size: int = 64,
        num_heads: int = 10,
        patch_grid_size: (int, int) = (16, 32),
        window_size: (int, int) = (8, 4),
    ) -> None:
        """Initialization for SWIN-Encoder block."""
        super().__init__()

        self.emb_size = emb_size
        self.num_heads = num_heads
        self.patch_grid_size = patch_grid_size
        self.window_size = window_size

        self.WMSA = WindowMSA(
            emb_size=self.emb_size,
            num_heads=self.num_heads,
            patch_grid_size=self.patch_grid_size,
            window_size=self.window_size,
        )
        self.SWMSA = ShiftedWindowMSA(
            emb_size=self.emb_size,
            num_heads=self.num_heads,
            patch_grid_size=self.patch_grid_size,
            window_size=self.window_size,
        )

        self.ln = nn.LayerNorm(self.emb_size)
        self.MLP = MLP(self.emb_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method for SWIN-Encoder block."""
        # Window Attention
        x = x + self.WMSA(self.ln(x))
        x = x + self.MLP(self.ln(x))
        # Shifted Window Attention
        x = x + self.SWMSA(self.ln(x))
        x = x + self.MLP(self.ln(x))

        return x


class SwinEncoder2(nn.Module):
    """The main SWIN-V2 encoder.

    The main SWIN-V2 encoder changes the original SWIN encoder to apply layer
    normalization after the MLP layers and MSA layers. The MSA layers are also
    modified to use a *cosine* self-attention mechanism with a learnable,
    per-head, scaling.

    Args:
        emb_size (int): Incoming embedding dimension.
        num_heads (int): Number of heads to use in the MSA.
        patch_grid_size (int, int): Grid dimensions making up the token list.
        window_size (int, int): Dimensions of window to use on the patch grid.

    """

    def __init__(
        self,
        emb_size: int = 64,
        num_heads: int = 10,
        patch_grid_size: (int, int) = (16, 32),
        window_size: (int, int) = (8, 4),
    ) -> None:
        """Initialization for SWIN-V2 Encoder block."""
        super().__init__()

        self.emb_size = emb_size
        self.num_heads = num_heads
        self.patch_grid_size = patch_grid_size
        self.window_size = window_size

        self.WMSA = WindowCosMSA(
            emb_size=self.emb_size,
            num_heads=self.num_heads,
            patch_grid_size=self.patch_grid_size,
            window_size=self.window_size,
        )
        self.SWMSA = ShiftedWindowCosMSA(
            emb_size=self.emb_size,
            num_heads=self.num_heads,
            patch_grid_size=self.patch_grid_size,
            window_size=self.window_size,
        )

        self.ln = nn.LayerNorm(self.emb_size)
        self.MLP = MLP(self.emb_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method for SWIN-V2 Encoder block."""
        # Window Attention
        x = x + self.ln(self.WMSA(x))
        x = x + self.ln(self.MLP(x))
        # Shifted Window Attention
        x = x + self.ln(self.SWMSA(x))
        x = x + self.ln(self.MLP(x))

        return x


class SwinConnectEncoder(SwinEncoder2):
    """SWIN-V2 Encoder with residuals.

    A SWIN-V2 encoder that outputs a copy of its forward pass to use in
    residual or skip connection layers.

    Args:
        emb_size (int): Incoming embedding dimension.
        num_heads (int): Number of heads to use in the MSA.
        patch_grid_size (int, int): Grid dimensions making up the token list.
        window_size (int, int): Dimensions of window to use on the patch grid.

    """

    def __init__(
        self,
        emb_size: int = 64,
        num_heads: int = 10,
        patch_grid_size: (int, int) = (16, 32),
        window_size: (int, int) = (8, 4),
    ) -> None:
        """Initialization for the SWIN-V2 encoder with residuals."""
        super().__init__(
            emb_size=emb_size,
            num_heads=num_heads,
            patch_grid_size=patch_grid_size,
            window_size=window_size,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward method for the SWIN-V2 encoder with residuals."""
        x = super().forward(x)

        return x, x


class SwinConnectDecoder(SwinEncoder2):
    """SWIN-V2 encoder with residual input.

    A SWIN-V2 encoder that appends an extra input tensor and then remaps to the
    embedding dimension through a linear layer prior to passing the result
    through a standard SWIN-V2 encoder.

    Args:
        emb_size (int): Incoming embedding dimension.
        num_heads (int): Number of heads to use in the MSA.
        patch_grid_size (int, int): Grid dimensions making up the token list.
        window_size (int, int): Dimensions of window to use on the patch grid.

    """

    def __init__(
        self,
        emb_size: int = 64,
        num_heads: int = 10,
        patch_grid_size: (int, int) = (16, 32),
        window_size: (int, int) = (8, 4),
    ) -> None:
        """Initialization for SWIN-V2 Connect Decoder."""
        super().__init__(
            emb_size=emb_size,
            num_heads=num_heads,
            patch_grid_size=patch_grid_size,
            window_size=window_size,
        )

        self.linear_remap = nn.Linear(2 * emb_size, emb_size)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward method for SWIN-V2 Connect Decoder."""
        # Concatenate with the skip connection input
        x = torch.cat([x, y], dim=-1)

        # Remap to embedding dimension linearly
        x = self.linear_remap(x)

        # Standard SWIN-V2 encoding
        x = super().forward(x)

        return x


if __name__ == "__main__":
    """Usage Example.

    """

    # Assume original image is (1120, 800) and embedded with
    # patch-size (20, 20).
    #
    # (B, token_number, E) = (3, 1024, 64)
    x = torch.rand(3, 56 * 40, 64)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = x.type(torch.FloatTensor).to(device)

    num_heads = 8
    emb_size = 64
    window_size = (8, 10)  # Due to shift each dimension must be divisible by 2.
    patch_grid_size = (56, 40)
    model_swin_encoder = SwinEncoder(
        emb_size=emb_size,
        num_heads=num_heads,
        patch_grid_size=patch_grid_size,
        window_size=window_size,
    ).to(device)
    model_swinV2_encoder = SwinEncoder2(
        emb_size=emb_size,
        num_heads=num_heads,
        patch_grid_size=patch_grid_size,
        window_size=window_size,
    ).to(device)
    model_swin_connect = SwinConnectEncoder(
        emb_size=emb_size,
        num_heads=num_heads,
        patch_grid_size=patch_grid_size,
        window_size=window_size,
    ).to(device)
    model_swin_decoder = SwinConnectDecoder(
        emb_size=emb_size,
        num_heads=num_heads,
        patch_grid_size=patch_grid_size,
        window_size=window_size,
    ).to(device)

    print("Input shape:", x.shape)
    print("SWIN encoder shape:", model_swin_encoder(x).shape)
    print("SWIN-V2 encoder shape:", model_swinV2_encoder(x).shape)
    x, y = model_swin_connect(x)
    print("SWIN connect encoder shape:", x.shape, y.shape)
    print("SWIN connect decoder shape:", model_swin_decoder(x, y).shape)
