"""Vector to Image CNN nn.Modules.

PyTorch nn.Module classes forming CNNs mapping vectors of scalar inputs to
images.

"""

from collections import OrderedDict

import torch
import torch.nn as nn

from yoke.torch_training_utils import count_torch_params


class jekelCNNsurrogate(nn.Module):
    """Vector-to-image CNN.

    Convolutional Neural Network Module that creates a scalar-to-image
    surrogate using a sequence of ConvTranspose2D, Batch Normalization, and
    Activation layers.

    This architecture is meant to reproduce the architecture described in
    Jekel et. al. 2022 *Using conservation laws to infer deep learning
    model accuracy of Richtmyer-Meshkov instabilities.*

    Args:
        input_size (int): Size of input
        linear_features (tuple[int, int]): Window size scalar parameters are
                                           originally mapped into
        kernel (tuple[int, int]): Size of transpose-convolutional kernel
        nfeature_list (list[int]): List of number of features in each
                                   T-convolutional layer
        output_image_size (tuple[int, int]): Image size to output, (H, W).
                                             Channels are automatically inherited.
        act_layer(nn.modules.activation): torch neural network layer class
                                          to use as activation

    """

    def __init__(
        self,
        input_size: int = 29,
        linear_features: tuple[int, int] = (4, 4),
        kernel: tuple[int, int] = (3, 3),
        nfeature_list: list[int] = [512, 512, 512, 512, 256, 128, 64, 32],
        output_image_size: tuple[int, int] = (1120, 800),
        act_layer: nn.Module = nn.GELU,
    ) -> None:
        """Initialization for Jekel t-CNN."""
        super().__init__()

        self.input_size = input_size
        self.output_image_size = output_image_size
        self.linear_features = linear_features
        self.nfeature_list = nfeature_list
        self.kernel = kernel
        self.nConvT = len(self.nfeature_list)

        # First linear remap
        out_features = (
            self.linear_features[0] * self.linear_features[1] * self.nfeature_list[0]
        )
        self.dense_expand = nn.Linear(
            in_features=self.input_size, out_features=out_features, bias=False
        )

        normLayer = nn.BatchNorm2d(self.nfeature_list[0])
        nn.init.constant_(normLayer.weight, 1)
        normLayer.weight.requires_grad = False

        self.inNorm = normLayer
        self.inActivation = act_layer()

        # Module list to hold transpose convolutions
        self.TConvList = nn.ModuleList()
        self.BnormList = nn.ModuleList()
        self.ActList = nn.ModuleList()

        # Create transpose convolutional layer for each entry in feature list.
        for i in range(self.nConvT - 1):
            tconv = nn.ConvTranspose2d(
                in_channels=self.nfeature_list[i],
                out_channels=self.nfeature_list[i + 1],
                kernel_size=self.kernel,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            )

            self.TConvList.append(tconv)

            normLayer = nn.BatchNorm2d(self.nfeature_list[i + 1])
            nn.init.constant_(normLayer.weight, 1)
            normLayer.weight.requires_grad = False

            self.BnormList.append(normLayer)
            self.ActList.append(act_layer())

        # Final Transpose Conv layer followed by hyperbolic tanh activation
        self.final_tconv = nn.ConvTranspose2d(
            in_channels=self.nfeature_list[-1],
            out_channels=1,
            kernel_size=self.kernel,
            stride=2,
            padding=1,
            output_padding=1,
            bias=True,
        )

        # If normalizing to [-1, 1]
        # self.final_act = nn.Tanh()

        # Else...
        self.final_act = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method for Jekel t-CNN."""
        # Input Layers
        x = self.dense_expand(x)
        # Reshape to a 2D block with channels
        # NOTE: -1 infers batch size
        x = x.view(
            -1, self.nfeature_list[0], self.linear_features[0], self.linear_features[1]
        )

        x = self.inNorm(x)
        x = self.inActivation(x)
        # print('After dense-map shape:', x.shape)

        # ConvT layers
        for i in range(self.nConvT - 1):
            x = self.TConvList[i](x)
            x = self.BnormList[i](x)
            x = self.ActList[i](x)
            # print(f'After convT{i:d} shape:', x.shape)

        # Final ConvT
        x = self.final_tconv(x)
        x = self.final_act(x)
        # print('After final convT shape:', x.shape)

        # Alternate resize
        x = nn.functional.interpolate(
            x, size=self.output_image_size, mode="bilinear", antialias=True
        )
        # print('Post-Upsample shape:', x.shape)

        return x


class tCNNsurrogate(nn.Module):
    """Vector-to-Image CNN.

    Convolutional Neural Network Module that creates a scalar-to-image
    surrogate using a sequence of ConvTranspose2D, Batch Normalization, and
    Activation layers.

    This architecture is meant to reproduce the architecture described in
    Jekel et. al. 2022 *Using conservation laws to infer deep learning
    model accuracy of Richtmyer-Meshkov instabilities.*

    However, image sizes are not always square powers of 2. Therefore, we
    allow a transpose convolution with specified parameters to resize the
    initial image stack to something that can be upsized to the output
    image size by multiplying by a power of 2. Unlike the jekelCNNsurrogate
    class which dealt with resizing by interpolation in the last layer. It
    is confusing because it is...

    WARNING!!!

    If the linear_features, intial convolution parameters, and feature list are
    not set up carefully then the output will be different than the expected
    output image size. A helper function should be constructed to aid in
    checking consistency but is not available now.

    WARNING!!!

    Args:
        input_size (int): Size of input
        linear_features (tuple[int, int, int]): Window size and number of features
                                                scalar parameters are originally
                                                mapped into
        initial_tconv_kernel (tuple[int, int]): Kernel size of initial tconv2d
        initial_tconv_stride (tuple[int, int]): Stride size of initial tconv2d
        initial_tconv_padding (tuple[int, int]): Padding size of initial tconv2d
        initial_tconv_outpadding (tuple[int, int]): Outout padding size of
                                                    initial tconv2d
        initial_tconv_dilation (tuple[int, int]): Dilation size of initial tconv2d
        kernel (tuple[int, int]): Size of transpose-convolutional kernel
        nfeature_list (list[int]): List of number of features in each
                                   T-convolutional layer
        output_image_size (tuple[int, int]): Image size to output, (H, W).
        output_image_channels (int): Number of output image channels.
        act_layer(nn.modules.activation): torch neural network layer class
                                          to use as activation

    """

    def __init__(
        self,
        input_size: int = 29,
        linear_features: tuple[int, int] = (7, 5, 256),
        initial_tconv_kernel: tuple[int, int] = (5, 5),
        initial_tconv_stride: tuple[int, int] = (5, 5),
        initial_tconv_padding: tuple[int, int] = (0, 0),
        initial_tconv_outpadding: tuple[int, int] = (0, 0),
        initial_tconv_dilation: tuple[int, int] = (1, 1),
        kernel: tuple[int, int] = (3, 3),
        nfeature_list: list[int] = [256, 128, 64, 32, 16],
        output_image_size: tuple[int, int] = (1120, 800),
        output_image_channels: int = 1,
        act_layer: nn.Module = nn.GELU,
    ) -> None:
        """Initialization for the t-CNN surrogate."""
        super().__init__()

        self.input_size = input_size
        self.output_image_size = output_image_size  # This argument is not used currently
        self.output_image_channels = output_image_channels
        self.linear_features = linear_features
        self.initial_tconv_kernel = initial_tconv_kernel
        self.initial_tconv_stride = initial_tconv_stride
        self.initial_tconv_padding = initial_tconv_padding
        self.initial_tconv_outpadding = initial_tconv_outpadding
        self.initial_tconv_dilation = initial_tconv_dilation
        self.nfeature_list = nfeature_list
        self.kernel = kernel
        self.nConvT = len(self.nfeature_list)

        # First linear remap
        out_features = (
            self.linear_features[0] * self.linear_features[1] * self.linear_features[2]
        )
        self.dense_expand = nn.Linear(
            in_features=self.input_size, out_features=out_features, bias=False
        )

        normLayer = nn.BatchNorm2d(self.linear_features[2])
        nn.init.constant_(normLayer.weight, 1)
        normLayer.weight.requires_grad = False

        self.inNorm = normLayer
        self.inActivation = act_layer()

        # Initial tconv2d layer to prepare for doubling layers
        self.initTConv = nn.ConvTranspose2d(
            in_channels=self.linear_features[2],
            out_channels=self.nfeature_list[0],
            kernel_size=self.initial_tconv_kernel,
            stride=self.initial_tconv_stride,
            padding=self.initial_tconv_padding,
            output_padding=self.initial_tconv_outpadding,
            dilation=self.initial_tconv_dilation,
            bias=False,
        )

        normLayer = nn.BatchNorm2d(self.linear_features[2])
        nn.init.constant_(normLayer.weight, 1)
        normLayer.weight.requires_grad = False

        self.initTconvNorm = normLayer
        self.initTconvActivation = act_layer()

        # Module list to hold transpose convolutions
        self.CompoundConvTList = nn.ModuleList()
        # Create transpose convolutional layer for each entry in feature list.
        for i in range(self.nConvT - 1):
            tconv = nn.ConvTranspose2d(
                in_channels=self.nfeature_list[i],
                out_channels=self.nfeature_list[i + 1],
                kernel_size=self.kernel,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            )

            normLayer = nn.BatchNorm2d(self.nfeature_list[i + 1])
            nn.init.constant_(normLayer.weight, 1)
            normLayer.weight.requires_grad = False

            # Make list of small sequential modules. Then we'll use enumerate
            # in forward method.
            cmpd_dict = OrderedDict(
                [
                    (f"tconv{i:02d}", tconv),
                    (f"bnorm{i:02d}", normLayer),
                    (f"act{i:02d}", act_layer()),
                ]
            )
            self.CompoundConvTList.append(nn.Sequential(cmpd_dict))

        # Final Transpose Conv layer followed by hyperbolic tanh activation
        self.final_tconv = nn.ConvTranspose2d(
            in_channels=self.nfeature_list[-1],
            out_channels=self.output_image_channels,
            kernel_size=self.kernel,
            stride=2,
            padding=1,
            output_padding=1,
            bias=True,
        )

        # If normalizing to [-1, 1]
        # self.final_act = nn.Tanh()

        # Else...
        self.final_act = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method for the t-CNN surrogate."""
        # Input Layers
        x = self.dense_expand(x)
        # Reshape to a 2D block with channels
        # NOTE: -1 infers batch size
        x = x.view(
            -1, self.linear_features[2], self.linear_features[0], self.linear_features[1]
        )

        x = self.inNorm(x)
        x = self.inActivation(x)
        # print('After dense-map shape:', x.shape)

        # Initial resize tConv layer
        x = self.initTConv(x)
        x = self.initTconvNorm(x)
        x = self.initTconvActivation(x)
        # print('After initTconv shape:', x.shape)

        # enumeration of nn.moduleList is supported under `torch.jit.script`
        for i, cmpdTconv in enumerate(self.CompoundConvTList):
            x = cmpdTconv(x)

        # Final ConvT
        x = self.final_tconv(x)
        x = self.final_act(x)
        # print('After final convT shape:', x.shape)

        return x


if __name__ == "__main__":
    """For testing and debugging.

    """

    # Excercise model setup
    # NOTE: Model takes (BatchSize, ScalarDims) tensor.
    scalar_input = torch.rand(4, 29)
    # jCNN = jekelCNNsurrogate(input_size=29,
    #                          linear_features=(4, 4),
    #                          kernel=(3, 3),
    #                          nfeature_list=[512, 512, 512, 512, 256, 128, 64, 32],
    #                          output_image_size=(1120, 800),
    #                          act_layer=nn.GELU)

    jCNN = tCNNsurrogate(
        input_size=29,
        # linear_features=(7, 5, 256),
        linear_features=(7, 5, 256),
        initial_tconv_kernel=(5, 5),
        initial_tconv_stride=(5, 5),
        initial_tconv_padding=(0, 0),
        initial_tconv_outpadding=(0, 0),
        initial_tconv_dilation=(1, 1),
        kernel=(3, 3),
        nfeature_list=[256, 128, 64, 32, 16],
        # nfeature_list=[512, 512, 256, 128, 64],
        output_image_size=(1120, 800),
        act_layer=nn.GELU,
    )
    jCNN.eval()
    jCNN_pred = jCNN(scalar_input)

    print("Input shape:", scalar_input.shape)
    print("Output shape:", jCNN_pred.shape)

    N_jCNN_param = count_torch_params(jCNN)
    print("Number of parameters:", N_jCNN_param)
