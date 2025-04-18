"""Vector-and-Image to Vector-and-Image CNNs.

PyTorch nn.Module classes forming CNNs mapping vectors and images to vectors
and images.

"""

from collections import OrderedDict

import torch
import torch.nn as nn

from yoke.torch_training_utils import count_torch_params

from yoke.models.CNNmodules import CNN_Interpretability_Module
from yoke.models.CNNmodules import CNN_Reduction_Module


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


class hybrid2vectorCNN(nn.Module):
    """Vector-and-Image to Vector-and-Image CNNs.

    Convolutional Neural Network Module that maps a triple (y, H1, H2) to a
    vector, R. Here, y is a 1D-tensor, H1 and H2 are 2D-tensors, and R is a
    1D-tensor. Each input is first processed through an independent branch
    before concatenation to a dense network.

    Args:
        img_size (tuple[int, int, int]): (C, H, W) dimensions of H1 and H2.
        input_vector_size (int): Size of input vector
        output_dim (int): Dimension of vector output.
        features (int): Number of output channels/features for each convolutional layer.
        depth (int): Number of convolutional layers in each image processing branch.
        kernel (int): Size of symmetric convolutional kernels
        img_embed_dim (int): Number of features in MLP output from image embeddings.
        vector_embed_dim (int): Number of features in MLP output from image embeddings.
        vector_feature_list (tuple[int, ...]): Number of features in each hidden layer
                                               of vector-MLP.
        output_feature_list (tuple[int, ...]): Number of features in each hidden layer
                                               of final/output-MLP.
        act_layer(nn.Module): torch neural network layer class to use as activation
        norm_layer(nn.Module): torch neural network layer class to use as normalization
                               between MLP layers.

    """

    def __init__(
        self,
        img_size: tuple[int, int, int] = (1, 1120, 400),
        input_vector_size: int = 28,
        output_dim: int = 1,
        features: int = 12,
        depth: int = 12,
        kernel: int = 3,
        img_embed_dim: int = 32,
        vector_embed_dim: int = 32,
        size_reduce_threshold: tuple[int, int] = (8, 8),
        vector_feature_list: tuple[int, ...] = (32, 32, 64, 64),
        output_feature_list: tuple[int, ...] = (64, 128, 128, 64),
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        """Initialization for hybrid CNN."""
        super().__init__()

        self.img_size = img_size
        _, H, W = self.img_size
        self.kernel = kernel
        self.features = features
        self.img_embed_dim = img_embed_dim
        self.vector_embed_dim = vector_embed_dim
        self.vector_feature_list = vector_feature_list
        self.output_feature_list = output_feature_list
        self.depth = depth
        self.size_reduce_threshold = size_reduce_threshold
        self.input_vector_size = input_vector_size
        self.output_dim = output_dim
        self.act_layer = act_layer
        self.norm_layer = norm_layer

        # CNN processing branch for H1
        self.interpH1 = CNN_Interpretability_Module(
            img_size=self.img_size,
            kernel=self.kernel,
            features=self.features,
            depth=self.depth,
            conv_onlyweights=True,
            batchnorm_onlybias=True,
            act_layer=self.act_layer,
        )

        self.reduceH1 = CNN_Reduction_Module(
            img_size=(self.features, H, W),
            size_threshold=self.size_reduce_threshold,
            kernel=self.kernel,
            stride=2,
            features=self.features,
            conv_onlyweights=True,
            batchnorm_onlybias=True,
            act_layer=self.act_layer,
        )

        self.finalW_h1 = self.reduceH1.finalW
        self.finalH_h1 = self.reduceH1.finalH

        # Linear embedding H1
        self.lin_embed_h1 = generalMLP(
            input_dim=self.finalH_h1 * self.finalW_h1 * self.features,
            output_dim=self.img_embed_dim,
            hidden_feature_list=(2 * self.img_embed_dim,),
            act_layer=self.act_layer,
            norm_layer=self.norm_layer,
        )

        # Image embed will end with a GELU activation
        self.h1_embed_act = self.act_layer()

        # CNN processing branch for H2
        self.interpH2 = CNN_Interpretability_Module(
            img_size=self.img_size,
            kernel=self.kernel,
            features=self.features,
            depth=self.depth,
            conv_onlyweights=True,
            batchnorm_onlybias=True,
            act_layer=self.act_layer,
        )

        self.reduceH2 = CNN_Reduction_Module(
            img_size=(self.features, H, W),
            size_threshold=self.size_reduce_threshold,
            kernel=self.kernel,
            stride=2,
            features=self.features,
            conv_onlyweights=True,
            batchnorm_onlybias=True,
            act_layer=self.act_layer,
        )

        self.finalW_h2 = self.reduceH2.finalW
        self.finalH_h2 = self.reduceH2.finalH

        # Linear embedding H2
        self.lin_embed_h2 = generalMLP(
            input_dim=self.finalH_h2 * self.finalW_h2 * self.features,
            output_dim=self.img_embed_dim,
            hidden_feature_list=(2 * self.img_embed_dim,),
            act_layer=self.act_layer,
            norm_layer=self.norm_layer,
        )

        # Image embed will end with a GELU activation
        self.h2_embed_act = self.act_layer()

        # MLP for processing vector input
        self.vector_mlp = generalMLP(
            input_dim=self.input_vector_size,
            output_dim=self.vector_embed_dim,
            hidden_feature_list=self.vector_feature_list,
            act_layer=self.act_layer,
            norm_layer=self.norm_layer,
        )

        self.vector_embed_act = self.act_layer()

        # Final MLP
        #
        # NOTE: Final activation is just identity.
        cat_size = self.vector_embed_dim + 2 * self.img_embed_dim
        self.final_mlp = generalMLP(
            input_dim=cat_size,
            output_dim=self.output_dim,
            hidden_feature_list=self.output_feature_list,
            act_layer=self.act_layer,
            norm_layer=nn.Identity,
        )

    def forward(
        self,
        y: torch.Tensor,
        h1: torch.Tensor,
        h2: torch.Tensor,
    ) -> torch.Tensor:
        """Forward method for hybrid CNN."""
        # Process first image
        h1_out = self.interpH1(h1)
        h1_out = self.reduceH1(h1_out)
        h1_out = torch.flatten(h1_out, start_dim=1)
        h1_out = self.lin_embed_h1(h1_out)
        h1_out = self.h2_embed_act(h1_out)

        # Process second image
        h2_out = self.interpH2(h2)
        h2_out = self.reduceH2(h2_out)
        h2_out = torch.flatten(h2_out, start_dim=1)
        h2_out = self.lin_embed_h2(h2_out)
        h2_out = self.h2_embed_act(h2_out)

        # Process vector
        y_out = self.vector_mlp(y)
        y_out = self.vector_embed_act(y_out)

        # Concatenate outputs and send through final MLP layer.
        cat = torch.cat((y_out, h1_out, h2_out), dim=1)
        out = self.final_mlp(cat)

        return out


if __name__ == "__main__":
    """For testing and debugging.

    """

    # Excercise model setup
    batch_size = 4
    img_h = 1120
    img_w = 400
    input_vector_size = 28
    output_dim = 5
    y = torch.rand(batch_size, input_vector_size)
    H1 = torch.rand(batch_size, 1, img_h, img_w)
    H2 = torch.rand(batch_size, 1, img_h, img_w)

    value_model = hybrid2vectorCNN(
        img_size=(1, img_h, img_w),
        input_vector_size=input_vector_size,
        output_dim=output_dim,
        features=12,
        img_embed_dim=32,
        vector_embed_dim=32,
        vector_feature_list=[32, 32, 64, 64],
        depth=12,
        kernel=3,
        size_reduce_threshold=(8, 8),
        act_layer=nn.GELU,
    )

    value_model.eval()
    value_pred = value_model(y, H1, H2)
    print("value_pred shape:", value_pred.shape)
    print(
        "Number of trainable parameters in value network:",
        count_torch_params(value_model, trainable=True),
    )
