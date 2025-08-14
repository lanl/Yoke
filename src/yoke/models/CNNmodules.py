"""Image to vector CNN modules."""

from collections import OrderedDict

import torch
import torch.nn as nn

from yoke.models.cnn_utils import conv2d_shape
from yoke.models.cnn_utils import generalMLP
from yoke.utils.parameters import count_torch_params


####################################
# Interpretability Module
####################################
class CNN_Interpretability_Module(nn.Module):
    """Interpretability module.

    Convolutional Neural Network Module that creates the "interpretability
    layers" Sequence of Conv2D, Batch Normalization, and Activation. The key
    idea is to keep the size of the image approximately equal throughout the
    network.

    Args:
        img_size (tuple[int, int, int]): size of input (channels, height, width)
        kernel (int): size of square convolutional kernel
        features (int): number of features in the convolutional layers
        depth (int): number of interpretability blocks
        conv_onlyweights (bool): determines if convolutional layers learn
                                 only weights or weights and bias
        batchnorm_onlybias (bool): determines if the batch normalization
                                   layers learn only bias or weights and bias
        act_layer(nn.modules.activation): torch neural network layer class
                                          to use as activation

    """

    def __init__(
        self,
        img_size: tuple[int, int, int] = (1, 1700, 500),
        kernel: int = 5,
        features: int = 12,
        depth: int = 12,
        conv_onlyweights: bool = True,
        batchnorm_onlybias: bool = True,
        act_layer: nn.Module = nn.GELU,
    ) -> None:
        """Initialization for interpretability CNN."""
        super().__init__()

        self.img_size = img_size
        C, _, _ = self.img_size
        self.kernel = kernel
        self.features = features
        self.depth = depth
        self.conv_weights = True
        self.conv_bias = not conv_onlyweights
        self.batchnorm_weights = not batchnorm_onlybias
        self.batchnorm_bias = True

        # Input Layers
        self.inConv = nn.Conv2d(
            in_channels=C,
            out_channels=self.features,
            kernel_size=self.kernel,
            stride=1,
            padding="same",  # pads the input so the output
            # has the shape as the input,
            # stride=1 only
            bias=self.conv_bias,
        )

        normLayer = nn.BatchNorm2d(features)

        if not self.batchnorm_weights:
            nn.init.constant_(normLayer.weight, 1)
            normLayer.weight.requires_grad = False

        self.inNorm = normLayer
        self.inActivation = act_layer()

        # Module list to hold interpretability layers
        self.InterpConvList = nn.ModuleList()
        for i in range(self.depth - 1):
            interpLayer = nn.Conv2d(
                in_channels=self.features,
                out_channels=self.features,
                kernel_size=self.kernel,
                stride=1,
                padding="same",  # pads the input so the
                # output has the shape of
                # the input, stride=1 only
                bias=self.conv_bias,
            )

            normLayer = nn.BatchNorm2d(features)

            # Necessary step to turn off only the scaling.
            if not self.batchnorm_weights:
                nn.init.constant_(normLayer.weight, 1)
                normLayer.weight.requires_grad = False

            # Make list of small sequential modules. Then we'll use enumerate
            # in forward method.
            interp_dict = OrderedDict(
                [
                    (f"interp{i:02d}", interpLayer),
                    (f"bnorm{i:02d}", normLayer),
                    (f"act{i:02d}", act_layer()),
                ]
            )
            self.InterpConvList.append(nn.Sequential(interp_dict))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method for interpretable CNN."""
        # Input Layers
        x = self.inConv(x)
        x = self.inNorm(x)
        x = self.inActivation(x)

        # Interpretability Layers
        for i, interp_conv in enumerate(self.InterpConvList):
            x = interp_conv(x)

        return x


####################################
# Reduction Module
####################################
class CNN_Reduction_Module(nn.Module):
    """Reduction CNN.

    Convolutional Neural Network Module that creates the "reduction layers"
    Sequence of Conv2D, Batch Normalization, and Activation. Key idea is to
    halve the image size at each layer using double-strided convolutions.

    Args:
        img_size (tuple[int, int, int]): size of input
                                         (channels, height, width)
        size_threshold (tuple[int, int]): (approximate) size of final,
                                          reduced image (height, width)
        kernel (int): size of square convolutional kernel
        stride (int): size of base stride for convolutional kernel
        features (int): number of features in the convolutional layers
        conv_onlyweights (bool): determines if convolutional layers learn
                                 only weights or weights and bias
        batchnorm_onlybias (bool): determines if the batch normalization layers
                                   learn only bias or weights and bias
        act_layer(nn.modules.activation): torch neural network layer class to
                                          use as activation

    """

    def __init__(
        self,
        img_size: tuple[int, int, int] = (1, 1700, 500),
        size_threshold: tuple[int, int] = (8, 8),
        kernel: int = 5,
        stride: int = 2,
        features: int = 12,
        conv_onlyweights: bool = True,
        batchnorm_onlybias: bool = True,
        act_layer: nn.Module = nn.GELU,
    ) -> None:
        """Initialization for reduction CNN."""
        super().__init__()

        self.img_size = img_size
        C, H, W = self.img_size
        self.size_threshold = size_threshold
        H_lim, W_lim = self.size_threshold
        self.kernel = kernel
        self.stride = stride
        self.features = features
        self.depth = 0  # initialize depth
        self.conv_weights = True
        self.conv_bias = not conv_onlyweights
        self.batchnorm_weights = not batchnorm_onlybias
        self.batchnorm_bias = True

        # Input Layers
        self.inConv = nn.Conv2d(
            in_channels=C,
            out_channels=self.features,
            kernel_size=self.kernel,
            stride=self.stride,
            padding=self.stride,
            padding_mode="zeros",
            bias=self.conv_bias,
        )

        normLayer = nn.BatchNorm2d(features)

        # Necessary step to turn off only the scaling.
        if not self.batchnorm_weights:
            nn.init.constant_(normLayer.weight, 1)
            normLayer.weight.requires_grad = False

        self.inNorm = normLayer
        self.inActivation = act_layer()

        W, H, _ = conv2d_shape(
            w=W,
            h=H,
            k=self.kernel,
            s_w=self.stride,
            s_h=self.stride,
            p_w=self.stride,
            p_h=self.stride,
        )

        self.depth += 1

        # Module list to hold reduction layers
        self.ReduceConvList = nn.ModuleList()
        while W > W_lim or H > H_lim:
            # Set Stride & Padding
            if W > W_lim:
                w_stride = self.stride
            else:
                w_stride = 1
            if H > H_lim:
                h_stride = self.stride
            else:
                h_stride = 1

            w_pad = 2 * w_stride
            h_pad = 2 * h_stride

            # Define Layers
            reduceLayer = nn.Conv2d(
                in_channels=self.features,
                out_channels=self.features,
                kernel_size=self.kernel,
                stride=(h_stride, w_stride),
                padding=(h_pad, w_pad),
                padding_mode="zeros",
                bias=self.conv_bias,
            )

            normLayer = nn.BatchNorm2d(features)

            if not self.batchnorm_weights:
                nn.init.constant_(normLayer.weight, 1)
                normLayer.weight.requires_grad = False

            # Make list of small sequential modules. Then we'll use enumerate
            # in forward method.
            self.depth += 1
            reduce_dict = OrderedDict(
                [
                    (f"reduce{self.depth:02d}", reduceLayer),
                    (f"bnorm{self.depth:02d}", normLayer),
                    (f"act{self.depth:02d}", act_layer()),
                ]
            )
            self.ReduceConvList.append(nn.Sequential(reduce_dict))

            # Recalculate Size
            W, H, _ = conv2d_shape(
                w=W, h=H, k=self.kernel, s_w=w_stride, s_h=h_stride, p_w=w_pad, p_h=h_pad
            )

        # Define final size
        self.finalW = W
        self.finalH = H

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method for reduction CNN."""
        # Input Layers
        x = self.inConv(x)
        x = self.inNorm(x)
        x = self.inActivation(x)

        # Reduction Layers
        for i, reduce_conv in enumerate(self.ReduceConvList):
            x = reduce_conv(x)

        return x


class PVI_SingleField_CNN(nn.Module):
    """Image to scalar CNN.

    The name of this class is a bit misleading, as it is a general CNN that maps
    an image to a scalar. The PVI task is too specific.

    Convolutional Neural Network Model that uses a single PVI field to predict
    one scalar value. Constructed using both an interpretability block, defined
    above, and a reduction block.

    Args:
        img_size (tuple[int, int, int]): size of input (channels, height, width)
        size_threshold (tuple[int, int]): (approximate) size of reduced image
                                          (height, width)
        kernel (int): size of square convolutional kernel
        features (int): number of features in the convolutional layers
        interp_depth (int): number of interpretability blocks
        conv_onlyweights (bool): determines if convolutional layers learn only
                                 weights or weights and bias
        batchnorm_onlybias (bool): determines if the batch normalization layers
                                   learn only bias or weights and bias
        act_layer (nn.modules.activation): torch neural network layer class to
                                           use as activation
        hidden_features (int): number of hidden features in the fully connected
                               dense layer

    """

    def __init__(
        self,
        img_size: tuple[int, int, int] = (1, 1700, 500),
        size_threshold: tuple[int, int] = (8, 8),
        kernel: int = 5,
        features: int = 12,
        interp_depth: int = 12,
        conv_onlyweights: bool = True,
        batchnorm_onlybias: bool = True,
        act_layer: nn.Module = nn.GELU,
        hidden_features: int = 20,
    ) -> None:
        """Initialization for image-to-scalar CNN."""
        super().__init__()
        self.img_size = img_size
        _, H, W = self.img_size
        self.size_threshold = size_threshold
        self.kernel = kernel
        self.features = features
        self.interp_depth = interp_depth
        self.hidden_features = hidden_features

        self.conv_onlyweights = conv_onlyweights
        self.conv_weights = True
        self.conv_bias = not self.conv_onlyweights
        self.batchnorm_onlybias = batchnorm_onlybias
        self.batchnorm_weights = not self.batchnorm_onlybias
        self.batchnorm_bias = True

        self.interp_module = CNN_Interpretability_Module(
            img_size=self.img_size,
            kernel=self.kernel,
            features=self.features,
            depth=self.interp_depth,
            conv_onlyweights=self.conv_onlyweights,
            batchnorm_onlybias=self.batchnorm_onlybias,
            act_layer=act_layer,
        )

        self.reduction_module = CNN_Reduction_Module(
            img_size=(self.features, H, W),
            size_threshold=self.size_threshold,
            kernel=self.kernel,
            features=self.features,
            conv_onlyweights=self.conv_onlyweights,
            batchnorm_onlybias=self.batchnorm_onlybias,
            act_layer=act_layer,
        )

        self.reduction_depth = self.reduction_module.depth
        self.finalW = self.reduction_module.finalW
        self.finalH = self.reduction_module.finalH

        self.endConv = nn.Conv2d(
            in_channels=self.features,
            out_channels=self.features,
            kernel_size=self.kernel,
            stride=1,
            padding="same",  # pads the input so the output
            # has the shape as the input,
            # stride=1 only
            bias=self.conv_bias,
        )
        self.endConvActivation = act_layer()

        # Hidden Layer (Equivalent to a matched convolutional layer?)
        self.hidden = nn.Linear(
            self.finalH * self.finalW * self.features, self.hidden_features
        )
        self.hiddenActivation = act_layer()

        # Linear Output Layer
        self.linOut = nn.Linear(self.hidden_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method for image-to-scalar CNN."""
        x = self.interp_module(x)
        x = self.reduction_module(x)

        # Final Convolution
        x = self.endConv(x)
        x = self.endConvActivation(x)
        x = torch.flatten(x, start_dim=1)

        # Hidden Layer
        x = self.hidden(x)
        x = self.hiddenActivation(x)

        # Linear Output Layer
        x = self.linOut(x)
        x = torch.flatten(x, start_dim=1)

        return x


class Image2VectorCNN(nn.Module):
    """Image to vector CNN.

    Convolutional Neural Network Model that maps an image to a vector. Constructed using
    both an interpretability block, defined above, and a reduction block.

    Args:
        img_size (tuple[int, int, int]): size of input (channels, height, width)
        output_dim (int): size of output vector
        size_threshold (tuple[int, int]): (approximate) size of reduced image
                                          (height, width)
        kernel (int): size of square convolutional kernel
        features (int): number of features in the convolutional layers
        interp_depth (int): number of interpretability blocks
        conv_onlyweights (bool): determines if convolutional layers learn only
                                 weights or weights and bias
        batchnorm_onlybias (bool): determines if the batch normalization layers
                                   learn only bias or weights and bias
        act_layer (nn.modules.activation): torch neural network layer class to
                                           use as activation
        norm_layer (nn.modules.normalization): torch neural network layer class
        hidden_features (int): number of hidden features in the fully connected
                               dense layer

    """

    def __init__(
        self,
        img_size: tuple[int, int, int] = (1, 1120, 400),
        output_dim: int = 29,
        size_threshold: tuple[int, int] = (12, 12),
        kernel: int = 3,
        features: int = 16,
        interp_depth: int = 12,
        conv_onlyweights: bool = True,
        batchnorm_onlybias: bool = True,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        hidden_features: int = 32,
    ) -> None:
        """Initialization for image-to-vector CNN."""
        super().__init__()
        self.img_size = img_size
        _, H, W = self.img_size
        self.output_dim = output_dim
        self.size_threshold = size_threshold
        self.kernel = kernel
        self.features = features
        self.interp_depth = interp_depth
        self.hidden_features = hidden_features

        self.conv_onlyweights = conv_onlyweights
        self.conv_weights = True
        self.conv_bias = not self.conv_onlyweights
        self.batchnorm_onlybias = batchnorm_onlybias
        self.batchnorm_weights = not self.batchnorm_onlybias
        self.batchnorm_bias = True

        self.interp_module = CNN_Interpretability_Module(
            img_size=self.img_size,
            kernel=self.kernel,
            features=self.features,
            depth=self.interp_depth,
            conv_onlyweights=self.conv_onlyweights,
            batchnorm_onlybias=self.batchnorm_onlybias,
            act_layer=act_layer,
        )

        self.reduction_module = CNN_Reduction_Module(
            img_size=(self.features, H, W),
            size_threshold=self.size_threshold,
            kernel=self.kernel,
            features=self.features,
            conv_onlyweights=self.conv_onlyweights,
            batchnorm_onlybias=self.batchnorm_onlybias,
            act_layer=act_layer,
        )

        self.reduction_depth = self.reduction_module.depth
        self.finalW = self.reduction_module.finalW
        self.finalH = self.reduction_module.finalH

        self.endConv = nn.Conv2d(
            in_channels=self.features,
            out_channels=self.features,
            kernel_size=self.kernel,
            stride=1,
            padding="same",  # pads the input so the output
            # has the shape as the input,
            # stride=1 only
            bias=self.conv_bias,
        )
        self.endConvActivation = act_layer()

        # Hidden Layer (Equivalent to a matched convolutional layer?)
        self.hidden = nn.Linear(
            self.finalH * self.finalW * self.features, self.hidden_features
        )
        self.hiddenActivation = act_layer()

        # MLP Output Layer
        self.out_mlp = generalMLP(
            input_dim=self.hidden_features,
            output_dim=self.output_dim,
            hidden_feature_list=(2 * self.hidden_features,),
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method for image-to-scalar CNN."""
        x = self.interp_module(x)
        x = self.reduction_module(x)

        # Final Convolution
        x = self.endConv(x)
        x = self.endConvActivation(x)
        x = torch.flatten(x, start_dim=1)

        # Hidden Layer
        x = self.hidden(x)
        x = self.hiddenActivation(x)

        # MLP Output Layer
        x = self.out_mlp(x)

        return x


if __name__ == "__main__":
    """For testing and debugging.

    """

    # Excercise model setup
    # NOTE: Model takes (BatchSize, Channels, Height, Width) tensor.
    pvi_input = torch.rand(1, 1, 1700, 500)
    pvi_CNN = PVI_SingleField_CNN(
        img_size=(1, 1700, 500),
        size_threshold=(8, 8),
        kernel=5,
        features=12,
        interp_depth=15,
        conv_onlyweights=True,
        batchnorm_onlybias=True,
        act_layer=nn.GELU,
        hidden_features=20,
    )

    pvi_CNN.eval()
    pvi_pred = pvi_CNN(pvi_input)

    # Excercise model setup
    batch_size = 2
    img_h = 1120
    img_w = 800
    output_dim = 29
    Hfield = torch.rand(batch_size, 1, img_h, img_w)

    model_args = {
        "img_size": (1, img_h, img_w),
        "output_dim": output_dim,
        "size_threshold": (12, 12),
        "kernel": 3,
        "features": 16,
        "interp_depth": 12,
        "conv_onlyweights": True,
        "batchnorm_onlybias": True,
        "act_layer": nn.GELU,
        "norm_layer": nn.LayerNorm,
        "hidden_features": 32,
    }
    img2vec_CNN = Image2VectorCNN(**model_args)
    img2vec_CNN.eval()
    y_pred = img2vec_CNN(Hfield)
    print("y_pred shape:", y_pred.shape)
    print(
        "Number of trainable parameters in img2vec_CNN network:",
        count_torch_params(img2vec_CNN, trainable=True),
    )