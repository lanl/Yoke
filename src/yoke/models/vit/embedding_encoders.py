"""Various embeddings for ViT networks.

Module containing classes for *variable embeddings*, *position embeddings*,
*temporal embeddings*.

"""

import numpy as np

import torch
import torch.nn as nn


def get_1d_sincos_pos_embed_from_grid(
    embed_dim: int, position: np.ndarray
) -> np.ndarray:
    """1D Sine/Cosine embedding.

    Args:
        embed_dim (int): Must be divisible by 2. Output dimension, D, for each
                         position
        position (np.array): Size (M, 1) array of 1D-positions
                             to be encoded.

    Returns:
        emb (np.array): Size (M, D) array consisting of the outer product of
                        position-index values and a phase passed through sine
                        and cosine. The first D/2 columns are passed through
                        Sine, the second D/2 columns are passed through Cosine.

    """
    assert embed_dim % 2 == 0

    # omega = 10000^(-w) with w=(2/D)*[0, 1, 2, ..., (D/2) - 1]
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    # Flatten position array
    position = position.reshape(-1)  # (M,)

    # Compute outer product of position vector and omega vector:
    #
    # out = position(M, 1) x omega(1, D/2) or out(i, j) = position(i) x omega(j)
    out = np.einsum("m,d->md", position, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)

    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    """2D Sine/Cosine embedding using grid array.

    Args:
        embed_dim (int): Must be divisible by 2. Output dimension, D, for each
                         position
        grid (np.array): Size (2, *singleton dimensions*, M) array of 2D-positions
                         to be encoded. Singleton dimensions are ignored through
                         flattening of the grid[0] and grid[1] sub-arrays.

    Returns:
        emb (np.array): Size (2, M, D) array with each (1, M, D)-subarray
                        consisting of the outer product of
                        position-index values and a phase passed through sine
                        and cosine. The first D/2 columns are passed through
                        Sine, the second D/2 columns are passed through Cosine.

    """
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (M, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (M, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (M, D)

    return emb


def get_2d_sincos_pos_embed(
    embed_dim: int, grid_size_h: int, grid_size_w: int, cls_token: bool = False
) -> np.ndarray:
    """2D Sine/Cosine embedding on a index-grid of specified dimensions.

    Args:
        embed_dim (int): Must be divisible by 2. Output dimension, D, for each
                         position
        grid_size_h (int): Index grid height.
        grid_size_w (int): Index grid width.
        cls_token (bool): Whether or not a class-token embedding is appended.

    Returns:
        pos_embed (np.array): [grid_size_h*grid_size_w, embed_dim] OR
                              [1+grid_size*grid_size, embed_dim] with class token

    """
    assert embed_dim % 2 == 0

    W = grid_size_w
    H = grid_size_h
    grid_h = np.arange(H, dtype=np.float32)
    grid_w = np.arange(W, dtype=np.float32)
    # Create list of arrays for each grid dimension.
    # grid[0] = repeat(row(0:grid_w))
    # grid[1] = repeat(col(0:grid_h))
    grid = np.meshgrid(grid_w, grid_h)  # grid[i].shape=(W, H), i=1,2
    grid = np.stack(grid, axis=0)  # (2, W, H)

    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)

    return pos_embed


class VarEmbed(nn.Module):
    """Variable encoding/embedding.

    Encodes information regarding which variable each token belongs to. Helps
    in tracking variables through variable aggregation layer. Prior to variable
    aggregation, embedding is added to parallel patch embedding output to tag
    variable entries.

    This embedding consists of learnable weights but the weights are
    initialized with a sin/cos embedding.

    Args:
        default_vars (list): list of default variables to be used for training
        embed_dim (int): Embedding dimension

    """

    def __init__(self, default_vars: list[str], embed_dim: int) -> None:
        """Initialization for Variable Embedding."""
        super().__init__()

        self.default_vars = default_vars
        self.embed_dim = embed_dim

        # var_map is dictionary {var_name_str : var_idx_int}
        #
        # var_embed is (1, max_vars, embed_dim) tensor
        self.var_embed, self.var_map = self.create_var_embedding(embed_dim)

        # Initialize var_embed with sincos embedding
        self.initialize_weights()

    def create_var_embedding(self, dim: int) -> tuple[torch.Tensor, dict]:
        """Create variable embedding.

        Returns a variable embedding tensor of the correct size along with a
        dictionary mapping variable name strings to integers that can be used
        to construct the `in_vars` input to the forward method during training.

        """
        var_embed = nn.Parameter(
            torch.zeros(1, len(self.default_vars), dim), requires_grad=True
        )
        # TODO: create a mapping from var --> idx
        var_map = {}
        idx = 0

        for var in self.default_vars:
            var_map[var] = idx
            idx += 1

        return var_embed, var_map

    def initialize_weights(self) -> None:
        """Weight initialization for variable embedding."""
        var_embed = get_1d_sincos_pos_embed_from_grid(
            self.var_embed.shape[-1], np.arange(len(self.default_vars))
        )
        self.var_embed.data.copy_(torch.from_numpy(var_embed).float().unsqueeze(0))

    # @lru_cache(maxsize=None)
    # def get_var_ids(self, vars, device):
    #     ids = np.array([self.var_map[var] for var in vars])
    #     return torch.from_numpy(ids).to(device)

    # def get_var_emb(self, var_emb, vars):
    #     ids = self.get_var_ids(vars, var_emb.device)
    #     return var_emb[:, ids, :]

    def forward(self, x: torch.Tensor, in_vars: torch.Tensor) -> torch.Tensor:
        """Forward method for variable embedding.

        NOTE: `in_vars` should be a (1, C) tensor of integers where the
        integers correspond to the variables present in the channels of
        `x`. This implies that every sample in the batch represented by `x`
        must correspond to the same variables.

        """
        # The input tensor is shape:
        #  (B, V, L, D)=(B, NumVars, NumTokens, embed_dim)

        # Variable embedding is (1, V, D)
        var_embed = self.var_embed[:, in_vars, :]
        # var_embed = self.get_var_emb(self.var_embed, vars)

        # Unsqueeze a dimension after variable dimension...
        #   var_embed -> (1, V, 1, D)
        x = x + var_embed.unsqueeze(2)  # B, V, L, D

        return x


class PosEmbed(nn.Module):
    """Position encoding/embedding.

    Position embedding is used to help track where each patch embedding is
    located in relation to the others. After variable aggregation, embedding is
    added to patch tokens. It does not encode information about absolute
    position of a patch. This makes this position embedding non-Unit-Aware.

    This embedding consists of learnable weights but the weights are
    initialized with a 2D-sine/cosine embedding.

    NOTE: Entries for image_size and patch_size should divide eachother evenly.

    Args:
        embed_dim (int): Embedding dimension, must be divisible by 2.
        patch_size (tuple): Height and width of patches
        image_size (tuple): Height and width of original image
        num_patches (int): Number of patches after initial patch embedding.

    """

    def __init__(
        self,
        embed_dim: int,
        patch_size: tuple[int, int],
        image_size: tuple[int, int],
        num_patches: int,
    ) -> None:
        """Initialization for position embedding."""
        super().__init__()

        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_patches = num_patches

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, self.embed_dim), requires_grad=True
        )

        self.initialize_weights()

    def initialize_weights(self) -> None:
        """Initialize position embedding weights using Sine/Cosine grid."""
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.image_size[0] / self.patch_size[0]),
            int(self.image_size[1] / self.patch_size[1]),
            cls_token=False,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method for position embedding."""
        # The input tensor is shape:
        #  (B, L, D)=(B, NumTokens[i.e. Patches], embed_dim)

        x = x + self.pos_embed

        return x


class RelativePositionEmbed(nn.Module):
    """Relative spatial encoding/embedding in SWIN.

    Relative spatial encoding/embedding used in the SWIN windowed multi-headed
    self-attention layers. In this specialized version we set up a relative
    position embedding for a rectangular image.


    Args:
        window_size (int, int): Embedding dimension, must be divisible by 2.

    """

    def __init__(self, window_size: (int, int) = (8, 4)) -> None:
        """Initialization for Relative Position Embedding."""
        super().__init__()

        # Set window size
        self.window_size = window_size

        # Since these weights are an nn.Parameter they will be updated during
        # training.
        self.pos_embeddings = nn.Parameter(
            torch.randn(2 * self.window_size[0] - 1, 2 * self.window_size[1] - 1)
        )

        # Create array of indices for each point in a window of size
        # (wh, ww), 'indices.shape = (wh*ww, 2)'
        idx_array = np.array(
            [
                [x, y]
                for x in range(self.window_size[0])
                for y in range(self.window_size[1])
            ]
        )
        self.indices = torch.tensor(idx_array)

        # Using 'None' creates an extra singleton dimension,
        # i.e. self.indices[None, :, :] has dimension (1, wh*ww, 2).
        # Therefore, self.relative_indices has size (wh*ww, wh*ww, 2) and
        # the entry self.relative_indices[i, j, :, :] is the (delta-rowIDX, delta-colIDX)
        # differences between the i and j points in the window.
        self.relative_indices = self.indices[None, :, :] - self.indices[:, None, :]

        # The previous range of self.relative_indices entries was [-wh+1, wh-1]
        # for the row-indices and [-ww+1, ww-1] for the column-indices. By
        # adding wh-1 to each entry in the first channel and ww-1 to each entry
        # in the second channel the new ranges are [0, (2*wh-1)-1] and [0,
        # (2*ww-1)-1], respectively. This ensures the `relative_indices` array
        # can be used as indices in self.pos_embeddings
        self.relative_indices[:, :, 0] += self.window_size[0] - 1
        self.relative_indices[:, :, 1] += self.window_size[1] - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method for Relative Position Embedding."""
        # The input tensor is shape:
        #  (B, H, Hw, Ww, wh*ww, wh*ww)
        #
        # Here:
        #    B = batch size
        #    H = number of heads
        #    Hw = Height of window grid
        #    Ww = Width of window grid
        #    wh = window height
        #    ww = window_width
        #
        # In WindowedMSA, the input `x` represents the attention weights between
        # every pair of patches/tokens within each of the Hw x Ww
        # windows. There are wh*ww patches/tokens within each window so (wh*ww,
        # wh*ww) pairs so shape(x) = (B, H, Hw, Ww, wh*ww, wh*ww)

        # Get embedding weights between each pair of window positions.
        rel_pos_embedding = self.pos_embeddings[
            self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]
        ]

        # Using broadcasting the relative position embedding weights are added
        # to the last two dimensions of the attention weights for each batch,
        # head, and window.
        x = x + rel_pos_embedding

        return x


class TimeEmbed(nn.Module):
    """Temporal encoding/embedding.

    This embedding is used to help track/tag each entry of a batch by it's
    corresponding lead time. After variable aggregation and position encoding,
    temporal encoding is added to patch tokens.

    This embedding consists of a single 1D linear embedding of lead-times per
    sample in the batch to the embedding dimension.

    NOTE: Entries for image_size and patch_size should divide eachother evenly.

    Args:
        embed_dim (int): Embedding dimension, must be divisible by 2.

    Foward method args:
        lead_times (torch.Tensor): lead_times.shape = (B,). Forecasting lead
                                   times of each element of the batch.

    """

    def __init__(self, embed_dim: int) -> None:
        """Initialization for temporal embedding."""
        super().__init__()

        self.lead_time_embed = nn.Linear(1, embed_dim)

    def forward(self, x: torch.Tensor, lead_times: torch.Tensor) -> torch.Tensor:
        """Forward method for temporal embedding."""
        # The input tensor is shape:
        #  (B, L, D)=(B, NumTokens, embed_dim)

        # Add lead time embedding
        lead_time_emb = self.lead_time_embed(lead_times.unsqueeze(-1))  # B, D
        lead_time_emb = lead_time_emb.unsqueeze(1)  # B, 1, D
        x = x + lead_time_emb  # B, L, D

        return x


if __name__ == "__main__":
    """Usage Example.

    """

    default_vars = [
        "cu_pressure",
        "cu_density",
        "cu_temperature",
        "ss_pressure",
        "ss_density",
        "ss_temperature",
        "r_vel",
        "z_vel",
    ]

    embed_dim = 72
    patch_size = (16, 16)
    image_size = (128, 128)
    num_patches = 64

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Original: (B, C, H, W) = (3, 5, 128, 128)
    # After parallel-patch embedding: (B, V, L, D)
    x = torch.rand(3, 4, 64, 72)
    x = x.type(torch.FloatTensor).to(device)

    # Indices correspond to
    # ["cu_density", "ss_density", "ss_temperature", "r_vel"]
    in_vars = torch.tensor([1, 4, 5, 6]).to(device)

    # Prior to variable aggregation: (B, V, L, D)
    var_emb_model = VarEmbed(default_vars=default_vars, embed_dim=embed_dim).to(device)

    print("Vaiable Encoding Input shape:", x.shape)
    print(
        "Variable Encoding shape:",
        var_emb_model(x, in_vars=in_vars).shape,
    )

    # After variable aggregation: (B, L, D)
    x = torch.rand(3, 64, 72).to(device)
    pos_emb_model = PosEmbed(
        embed_dim=embed_dim,
        patch_size=patch_size,
        image_size=image_size,
        num_patches=num_patches,
    ).to(device)

    print("Position Encoding input shape:", x.shape)
    print("Position Encoding shape:", pos_emb_model(x).shape)

    # After position encoding: (B, L, D)
    x = torch.rand(3, 64, 72).to(device)
    lead_times = torch.rand(3).to(device)
    time_emb_model = TimeEmbed(embed_dim=embed_dim).to(device)

    print("Temporal Encoding input shape:", x.shape)
    print("Temporal Encoding shape:", time_emb_model(x, lead_times).shape)

    # Relative window embedding
    wh = 8
    ww = 5
    rel_emb_model = RelativePositionEmbed(window_size=(wh, ww)).to(device)
    print("Size of relative indices:", rel_emb_model.relative_indices.shape)
    print(
        "Minimum of x-coord relative indices:",
        rel_emb_model.relative_indices[:, :, 0].min(),
    )
    print(
        "Maximum of x-coord relative indices:",
        rel_emb_model.relative_indices[:, :, 0].max(),
    )
    print(
        "Minimum of y-coord relative indices:",
        rel_emb_model.relative_indices[:, :, 1].min(),
    )
    print(
        "Maximum of y-coord relative indices:",
        rel_emb_model.relative_indices[:, :, 1].max(),
    )

    # The input tensor is shape:
    #  (B, H, Hw, Ww, wh*ww, wh*ww)
    win_patches = torch.rand(3, 10, 8, 16, wh * ww, wh * ww).to(device)
    print("Size of relative-position embedding input:", win_patches.shape)
    print(
        "Size of relative-position embedding output:", rel_emb_model(win_patches).shape
    )
