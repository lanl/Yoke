"""Probabilistic CNN modules for RL policy networks."""

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

from yoke.torch_training_utils import count_torch_params

from yoke.models.CNNmodules import CNN_Interpretability_Module
from yoke.models.CNNmodules import CNN_Reduction_Module
from yoke.models.hybridCNNmodules import generalMLP


class gaussian_policyCNN(nn.Module):
    """Vector-and-Image to Gaussian distribution.

    Convolutional Neural Network Module that maps a triple (y, H1, H2) to a
    Gaussian distribution, N(x, C). Here, y is a 1D-tensor, H1 and H2 are
    2D-tensors. The mean x is a 1D-tensor and C is a 2D-tensor satisfying the
    symmetry and positive-definite properties of a covariance.

    Each input is first processed through an independent branch before
    concatenation to two forks of dense networks to estimate the mean and
    covariance.

    Args:
        img_size (tuple[int, int, int]): (C, H, W) dimensions of H1 and H2.
        input_vector_size (int): Size of input vector
        output_dim (int): Dimension of Guassian mean.
        min_variance (float): Minimum variance in diagonal covariance entries.
        features (int): Number of output channels/features for each convolutional layer.
        depth (int): Number of convolutional layers in each image processing branch.
        kernel (int): Size of symmetric convolutional kernels
        img_embed_dim (int): Number of features in MLP output from image embeddings.
        vector_embed_dim (int): Number of features in MLP output from image embeddings.
        vector_feature_list (tuple[int, ...]): Number of features in each hidden layer
                                               of vector-MLP.
        output_feature_list (tuple[int, ...]): Number of features in each hidden layer
                                               of final/output-MLP.
        act_layer (nn.Module): torch neural network layer class to use as activation
        norm_layer (nn.Module): torch neural network layer class to use as normalization
                                between MLP layers.

    """

    def __init__(
        self,
        img_size: tuple[int, int, int] = (1, 1120, 400),
        input_vector_size: int = 28,
        output_dim: int = 28,
        min_variance: float = 1e-6,
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
        self.min_variance = min_variance
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

        # Mean MLP
        #
        # NOTE: Final activation is just identity.
        cat_size = self.vector_embed_dim + 2 * self.img_embed_dim
        self.mean_mlp = generalMLP(
            input_dim=cat_size,
            output_dim=self.output_dim,
            hidden_feature_list=self.output_feature_list,
            act_layer=self.act_layer,
            norm_layer=nn.Identity,
        )

        # Covariance MLP
        self.num_cov_elements = self.output_dim * (self.output_dim + 1) // 2
        self.cov_mlp = generalMLP(
            input_dim=cat_size,
            output_dim=self.num_cov_elements,
            hidden_feature_list=self.output_feature_list,
            act_layer=self.act_layer,
            norm_layer=nn.Identity,
        )
        self._init_cov_mlp(self.cov_mlp, self.output_dim, self.min_variance)

    def _init_cov_mlp(
        self, mlp: nn.Module, output_dim: int, min_var: float = 1e-6
    ) -> None:
        """Initialize covariance layer to output identity.

        Initializes the final layer of the MLP such that the predicted Cholesky
        factor L results in a covariance matrix close to identity.

        Args:
            mlp (nn.Module): Multi-layer perceptron module. Must have last layer linear.
            output_dim (int): Dimension of network output.
            min_var (float): Minimum variance on covariance diagonal

        """
        tril_indices = torch.tril_indices(output_dim, output_dim)

        # Find the last linear layer (which has no activation)
        last_layer = mlp.LayerList[-1][0]  # Should be the "linear" layer
        err_msg = "Expected final MLP layer to be nn.Linear"
        assert isinstance(last_layer, nn.Linear), err_msg

        with torch.no_grad():
            # Start with all zeros
            last_layer.weight.zero_()
            last_layer.bias.zero_()

            # Set diagonal entries of L such that softplus(bias) + min_var ~ 1
            target_diag = 1.0 - min_var
            # softplus inverse
            init_bias_val = torch.log(torch.exp(torch.tensor(target_diag)) - 1.0)
            init_bias_val = init_bias_val.item()

            for idx, (row, col) in enumerate(zip(tril_indices[0], tril_indices[1])):
                if row == col:
                    last_layer.bias[idx] = init_bias_val  # Diagonal element
                else:
                    last_layer.bias[idx] = 0.0  # Off-diagonal

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
        h1_out = self.h1_embed_act(h1_out)

        # Process second image
        h2_out = self.interpH2(h2)
        h2_out = self.reduceH2(h2_out)
        h2_out = torch.flatten(h2_out, start_dim=1)
        h2_out = self.lin_embed_h2(h2_out)
        h2_out = self.h2_embed_act(h2_out)

        # Process vector
        y_out = self.vector_mlp(y)
        y_out = self.vector_embed_act(y_out)

        # Concatenate outputs and send on to mean and covariance prediction.
        cat = torch.cat((y_out, h1_out, h2_out), dim=1)

        # Predict the mean.
        mean = self.mean_mlp(cat)

        # Predict the covariance
        L_params = self.cov_mlp(cat)

        # Use Cholesky decomposition to ensure positive definite covariance.
        triIDXs = torch.tril_indices(self.output_dim, self.output_dim)
        L = torch.zeros(y.size(0), self.output_dim, self.output_dim, device=y.device)
        L[:, triIDXs[0], triIDXs[1]] = L_params

        # Ensure positive diagonal with softplus(z) = log(1+exp(z))
        diagIDXs = torch.arange(self.output_dim)
        L[:, diagIDXs, diagIDXs] = nn.functional.softplus(L[:, diagIDXs, diagIDXs])

        # Ensure a minimum variance
        L[:, diagIDXs, diagIDXs] += self.min_variance

        # Reconstruct covariance matrix assuming Cholesky factorization.
        cov_matrix = torch.matmul(L, L.transpose(-1, -2))

        # Return a MultivariateNormal distribution
        return MultivariateNormal(mean, covariance_matrix=cov_matrix)


if __name__ == "__main__":
    """For testing and debugging.

    """

    # Excercise model setup
    batch_size = 2
    img_h = 1120
    img_w = 800
    input_vector_size = 28
    output_dim = 28
    y = torch.rand(batch_size, input_vector_size)
    H1 = torch.rand(batch_size, 1, img_h, img_w)
    H2 = torch.rand(batch_size, 1, img_h, img_w)

    model_args_large = {
        "img_size": (1, img_h, img_w),
        "input_vector_size": input_vector_size,
        "output_dim": output_dim,
        "min_variance": 1e-6,
        "features": 12,
        "depth": 12,
        "kernel": 3,
        "img_embed_dim": 32,
        "vector_embed_dim": 32,
        "size_reduce_threshold": (8, 8),
        "vector_feature_list": (32, 32, 64, 64),
        "output_feature_list": (64, 128, 128, 64),
    }

    model_args_medium = {
        "img_size": (1, img_h, img_w),
        "input_vector_size": input_vector_size,
        "output_dim": output_dim,
        "min_variance": 1e-6,
        "features": 12,
        "depth": 15,
        "kernel": 3,
        "img_embed_dim": 32,
        "vector_embed_dim": 32,
        "size_reduce_threshold": (16, 16),
        "vector_feature_list": (16, 64, 64, 16),
        "output_feature_list": (16, 64, 64, 16),
    }

    model_args_small = {
        "img_size": (1, img_h, img_w),
        "input_vector_size": input_vector_size,
        "output_dim": output_dim,
        "min_variance": 1e-6,
        "features": 4,
        "depth": 6,
        "kernel": 3,
        "img_embed_dim": 16,
        "vector_embed_dim": 16,
        "size_reduce_threshold": (24, 24),
        "vector_feature_list": (8, 16, 16, 8),
        "output_feature_list": (8, 16, 16, 8),
    }

    policy_model = gaussian_policyCNN(**model_args_medium)

    policy_model.eval()
    policy_distribution = policy_model(y, H1, H2)
    print("Initial mean:", policy_distribution.mean)
    print("Initial covariance:", policy_distribution.covariance_matrix)
    print("Initial mean shape:", policy_distribution.mean.shape)
    print("Initial covariance shape:", policy_distribution.covariance_matrix.shape)
    print(
        "Number of trainable parameters in value network:",
        count_torch_params(policy_model, trainable=True),
    )
