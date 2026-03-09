"""Collection of helper functions to facilitate constructing CNN modules."""

import math

from collections import OrderedDict

import torch
import torch.nn as nn


####################################
# Get Conv2D Shape
####################################
def conv2d_shape(
    w: int, h: int, k: int, s_w: int, s_h: int, p_w: int, p_h: int
) -> tuple[int, int, int]:
    """Function to calculate the new dimension of an image after a nn.Conv2d.

    Args:
        w (int): starting width
        h (int): starting height
        k (int): kernel size
        s_w (int): stride size along the width
        s_h (int): stride size along the height
        p_w (int): padding size along the width
        p_h (int): padding size along the height

    Returns:
        new_w (int): number of pixels along the width
        new_h (int): number of pixels along the height
        total (int): total number of pixels in new image

    See Also:
    Formula taken from
    https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    Assuming a 2D input and dilation = 1

    """
    new_w = int(math.floor(((w + 2 * p_w - (k - 1) - 1) / s_w) + 1))
    new_h = int(math.floor(((h + 2 * p_h - (k - 1) - 1) / s_h) + 1))
    total = new_w * new_h

    return new_w, new_h, total


def convtranspose2d_shape(
    w: int,
    h: int,
    k_w: int,
    k_h: int,
    s_w: int,
    s_h: int,
    p_w: int,
    p_h: int,
    op_w: int,
    op_h: int,
    d_w: int,
    d_h: int,
) -> tuple[int, int, int]:
    """Calculate the dimension of an image after a nn.ConvTranspose2d.

    This assumes *groups*, *dilation*, and *ouput_padding* are all default
    values.

    Args:
        w (int): starting width
        h (int): starting height
        k_w (int): kernel width size
        k_h (int): kernel height size
        s_w (int): stride size along the width
        s_h (int): stride size along the height
        p_w (int): padding size along the width
        p_h (int): padding size along the height
        op_w (int): output padding size along the width
        op_h (int): output padding size along the height
        d_w (int): dilation size along the width
        d_h (int): dilation size along the height

    Returns:
        new_w (int): number of pixels along the width
        new_h (int): number of pixels along the height
        total (int): total number of pixels in new image

    See Also:
    Formula taken from
    https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html

    """
    new_w = (w - 1) * s_w - 2 * p_w + d_w * (k_w - 1) + op_w + 1
    new_h = (h - 1) * s_h - 2 * p_h + d_h * (k_h - 1) + op_h + 1
    total = new_w * new_h

    return new_w, new_h, total


class generalMLP(nn.Module):
    """A general multi-layer perceptron structure.

    Consists of stacked linear layers, normalizing layers, and
    activations. This is meant to be reused as a highly customizeable, but
    standardized, MLP structure.

    Args:
        input_dim (int): Dimension of input
        output_dim (int): Dimension of output
        hidden_feature_list (tuple[int, ...]): List of number of features in each layer.
                                               Length determines number of layers.
        act_layer (nn.modules.activation): torch neural network layer class to
                                           use as activation
        norm_layer (nn.Module): Normalization layer.

    """

    def __init__(
        self,
        input_dim: int = 64,
        output_dim: int = 16,
        hidden_feature_list: tuple[int, ...] = (16, 32, 32, 16),
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        """Initialization for MLP."""
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_feature_list = hidden_feature_list
        self.act_layer = act_layer
        self.norm_layer = norm_layer

        # Create full feature list without mutating input
        self.feature_list = (input_dim,) + hidden_feature_list + (output_dim,)

        # Module list to hold linear, normalization, and activation layers.
        self.LayerList = nn.ModuleList()
        # Create transpose convolutional layer for each entry in feature list.
        for i in range(len(self.feature_list) - 1):
            linear = nn.Linear(self.feature_list[i], self.feature_list[i + 1])

            normalize = self.norm_layer(self.feature_list[i + 1])
            activation = self.act_layer()

            # Make list of small sequential modules. Then we'll use enumerate
            # in forward method.
            #
            # Don't attach an activation to the final layer
            if i == len(self.feature_list) - 2:
                cmpd_dict = OrderedDict(
                    [
                        (f"linear{i:02d}", linear),
                    ]
                )
            else:
                cmpd_dict = OrderedDict(
                    [
                        (f"linear{i:02d}", linear),
                        (f"norm{i:02d}", normalize),
                        (f"act{i}", activation),
                    ]
                )

            self.LayerList.append(nn.Sequential(cmpd_dict))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method for MLP."""
        # enumeration of nn.moduleList is supported under `torch.jit.script`
        for i, ll_layer in enumerate(self.LayerList):
            x = ll_layer(x)

        return x


if __name__ == "__main__":
    """For testing and debugging.

    """

    # Excercise conv2d_shape function
    input_width = 100
    input_height = 150
    symmetric_kernel_size = 3
    stride_width = 2
    stride_height = 1
    padding_width = 2
    padding_height = 1
    new_w, new_h, total_pixels = conv2d_shape(
        input_width,
        input_height,
        symmetric_kernel_size,
        stride_width,
        stride_height,
        padding_width,
        padding_height,
    )
    print("New conv-image size:", new_w, new_h, total_pixels)

    kernel_width = 2
    kernel_height = 3
    new_w, new_h, total_pixels = convtranspose2d_shape(
        input_width,
        input_height,
        kernel_width,
        kernel_height,
        stride_width,
        stride_height,
        padding_width,
        padding_height,
    )
    print("New convtrans-image size:", new_w, new_h, total_pixels)
