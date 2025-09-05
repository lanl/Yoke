"""Test CNNmodule."""

import pytest
import torch
from torch import nn

from yoke.models.CNNmodules import CNN_Interpretability_Module
from yoke.models.CNNmodules import CNN_Reduction_Module
from yoke.models.CNNmodules import Image2ScalarCNN
from yoke.models.CNNmodules import Image2VectorCNN


###############################################################################
# Fixtures for CNN_Interpretability_Module
###############################################################################
@pytest.fixture
def default_interpretability_model() -> CNN_Interpretability_Module:
    """Pytest fixture for creating a default interpretability model."""
    return CNN_Interpretability_Module()


###############################################################################
# Tests for CNN_Interpretability_Module
###############################################################################
def test_default_forward_shape(
    default_interpretability_model: CNN_Interpretability_Module,
) -> None:
    """Test that the default interpretability model outputs the expected shape."""
    batch_size = 2
    c_in, height, width = default_interpretability_model.img_size
    x = torch.randn(batch_size, c_in, height, width)
    out = default_interpretability_model(x)
    assert out.shape == (
        batch_size,
        default_interpretability_model.features,
        height,
        width,
    )


def test_custom_model_forward_shape() -> None:
    """Test that a custom-configuration."""
    model = CNN_Interpretability_Module(
        img_size=(3, 224, 224),
        kernel=3,
        features=16,
        depth=4,
        conv_onlyweights=False,
        batchnorm_onlybias=False,
        act_layer=nn.ReLU,
    )
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 16, 224, 224)


def test_batchnorm_weights_frozen_interpretability() -> None:
    """Test that batchnorm weights are frozen if batchnorm_onlybias is True."""
    model = CNN_Interpretability_Module(batchnorm_onlybias=True)
    for name, param in model.named_parameters():
        if "Norm" in name and "weight" in name:
            assert not param.requires_grad


def test_batchnorm_weights_trainable_interpretability() -> None:
    """Test that batchnorm weights are trainable if batchnorm_onlybias is False."""
    model = CNN_Interpretability_Module(batchnorm_onlybias=False)
    for name, param in model.named_parameters():
        if "Norm" in name and "weight" in name:
            assert param.requires_grad


def test_conv_bias_toggle_interpretability() -> None:
    """Test that the convolutional bias toggles correctly for interpretability."""
    model_no_bias = CNN_Interpretability_Module(conv_onlyweights=True)
    model_with_bias = CNN_Interpretability_Module(conv_onlyweights=False)
    assert model_no_bias.inConv.bias is None
    assert model_with_bias.inConv.bias is not None


def test_forward_pass_no_exceptions_interpretability(
    default_interpretability_model: CNN_Interpretability_Module,
) -> None:
    """Test that the forward pass does not raise exceptions."""
    x = torch.randn(1, *default_interpretability_model.img_size)
    _ = default_interpretability_model(x)


def test_parameter_count_interpretability() -> None:
    """Test that parameter count for interpretability model is > 0."""
    model = CNN_Interpretability_Module()
    params = list(model.parameters())
    assert len(params) > 0


###############################################################################
# Fixtures for CNN_Reduction_Module
###############################################################################
@pytest.fixture
def default_reduction_model() -> CNN_Reduction_Module:
    """Pytest fixture for creating a default reduction model."""
    return CNN_Reduction_Module()


###############################################################################
# Tests for CNN_Reduction_Module
###############################################################################
def test_default_reduction_forward_shape(
    default_reduction_model: CNN_Reduction_Module,
) -> None:
    """Test that the default reduction model outputs a smaller (or equal) shape."""
    batch_size = 2
    c_in, h_in, w_in = default_reduction_model.img_size
    x = torch.randn(batch_size, c_in, h_in, w_in)
    out = default_reduction_model(x)
    assert out.shape[0] == batch_size
    assert out.shape[1] == default_reduction_model.features
    # Height/width should match finalH/finalW
    assert out.shape[2] == default_reduction_model.finalH
    assert out.shape[3] == default_reduction_model.finalW


def test_custom_reduction_forward_shape() -> None:
    """Test custom-configured reduction model."""
    model = CNN_Reduction_Module(
        img_size=(3, 128, 128),
        size_threshold=(16, 16),
        kernel=3,
        stride=2,
        features=8,
        conv_onlyweights=False,
        batchnorm_onlybias=False,
        act_layer=nn.ReLU,
    )
    x = torch.randn(2, 3, 128, 128)
    out = model(x)
    # Check channel count
    assert out.shape[1] == 8
    # Ensure final shape is <= (16, 16)
    assert out.shape[2] <= 16
    assert out.shape[3] <= 16


def test_batchnorm_weights_frozen_reduction() -> None:
    """Test that batchnorm weights are frozen if batchnorm_onlybias is True."""
    model = CNN_Reduction_Module(batchnorm_onlybias=True)
    for name, param in model.named_parameters():
        if "Norm" in name and "weight" in name:
            assert not param.requires_grad


def test_batchnorm_weights_trainable_reduction() -> None:
    """Test that batchnorm weights are trainable if batchnorm_onlybias is False."""
    model = CNN_Reduction_Module(batchnorm_onlybias=False)
    for name, param in model.named_parameters():
        if "Norm" in name and "weight" in name:
            assert param.requires_grad


def test_conv_bias_toggle_reduction() -> None:
    """Test that the convolutional bias toggles correctly for reduction module."""
    model_no_bias = CNN_Reduction_Module(conv_onlyweights=True)
    model_with_bias = CNN_Reduction_Module(conv_onlyweights=False)
    assert model_no_bias.inConv.bias is None
    assert model_with_bias.inConv.bias is not None


def test_forward_pass_no_exceptions_reduction(
    default_reduction_model: CNN_Reduction_Module,
) -> None:
    """Test forward pass of the reduction model."""
    x = torch.randn(1, *default_reduction_model.img_size)
    _ = default_reduction_model(x)


def test_parameter_count_reduction() -> None:
    """Test that parameter count for the reduction model is > 0."""
    model = CNN_Reduction_Module()
    params = list(model.parameters())
    assert len(params) > 0


###############################################################################
# Fixtures for Image2ScalarCNN
###############################################################################
@pytest.fixture
def default_image2scalar_model() -> Image2ScalarCNN:
    """Fixture for creating a default Image2ScalarCNN."""
    return Image2ScalarCNN()


###############################################################################
# Tests for Image2ScalarCNN
###############################################################################
def test_default_image2scalar_forward_shape(
    default_image2scalar_model: Image2ScalarCNN
) -> None:
    """Test that the default Image2ScalarCNN model produces a scalar output."""
    batch_size = 2
    c_in, height, width = default_image2scalar_model.img_size
    x = torch.randn(batch_size, c_in, height, width)
    out = default_image2scalar_model(x)
    # Should be [batch_size, 1]
    assert out.shape == (batch_size, 1)


def test_custom_image2scalar_forward_shape() -> None:
    """Test that a custom-configured Image2ScalarCNN model produces a scalar output."""
    model = Image2ScalarCNN(
        img_size=(3, 224, 224),
        size_threshold=(16, 16),
        kernel=3,
        features=8,
        interp_depth=3,
        conv_onlyweights=False,
        batchnorm_onlybias=False,
        act_layer=nn.ReLU,
        hidden_features=10,
    )
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 1)


def test_image2scalar_batchnorm_weights_frozen() -> None:
    """Test that batchnorm weights are frozen if batchnorm_onlybias=True."""
    model = Image2ScalarCNN(batchnorm_onlybias=True)
    for name, param in model.named_parameters():
        if "Norm" in name and "weight" in name:
            assert not param.requires_grad


def test_image2scalar_batchnorm_weights_trainable() -> None:
    """Test that batchnorm weights are trainable if batchnorm_onlybias=False."""
    model = Image2ScalarCNN(batchnorm_onlybias=False)
    for name, param in model.named_parameters():
        if "Norm" in name and "weight" in name:
            assert param.requires_grad


def test_image2scalar_conv_bias_toggle() -> None:
    """Test that the convolutional bias toggles correctly."""
    model_no_bias = Image2ScalarCNN(conv_onlyweights=True)
    model_with_bias = Image2ScalarCNN(conv_onlyweights=False)
    # Check the first convolution in the interpretability module
    assert model_no_bias.interp_module.inConv.bias is None
    assert model_with_bias.interp_module.inConv.bias is not None


def test_image2scalar_forward_pass_no_exceptions(
    default_image2scalar_model: Image2ScalarCNN
) -> None:
    """Test that the forward pass does not raise exceptions for Image2ScalarCNN model."""
    x = torch.randn(1, *default_image2scalar_model.img_size)
    _ = default_image2scalar_model(x)


def test_parameter_count_image2scalar() -> None:
    """Test that parameter count for the Image2ScalarCNN model is > 0."""
    model = Image2ScalarCNN()
    params = list(model.parameters())
    assert len(params) > 0


###############################################################################
# Fixtures for Image2VectorCNN
##############################################################################
@pytest.fixture
def small_image2vector_model() -> Image2VectorCNN:
    """Create a small, fast Image2VectorCNN instance for unit tests.

    Returns:
        An Image2VectorCNN configured with modest sizes to keep tests fast.
    """
    return Image2VectorCNN(
        img_size=(1, 64, 64),
        output_dim=5,
        size_threshold=(8, 8),
        kernel=3,
        features=8,
        interp_depth=2,
        conv_onlyweights=True,
        batchnorm_onlybias=True,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        hidden_features=16,
    )


###############################################################################
# Tests for Image2VectorCNN
###############################################################################
def test_image2vector_forward_shape(small_image2vector_model: Image2VectorCNN) -> None:
    """Validate that forward returns a (batch, output_dim) tensor."""
    bs = 2
    x = torch.randn(bs, *small_image2vector_model.img_size)
    y = small_image2vector_model.eval()(x)
    assert y.shape == (bs, small_image2vector_model.output_dim)


def test_image2vector_endconv_bias_flag() -> None:
    """Check conv bias flag mapping from conv_onlyweights parameter.

    When conv_onlyweights is True, convolutions should have no bias. When
    False, bias parameters should be present.
    """
    m_no_bias = Image2VectorCNN(img_size=(1, 32, 32), output_dim=3,
                                conv_onlyweights=True)
    assert m_no_bias.endConv.bias is None

    m_with_bias = Image2VectorCNN(img_size=(1, 32, 32), output_dim=3,
                                  conv_onlyweights=False)
    assert m_with_bias.endConv.bias is not None


def test_image2vector_batchnorm_freeze_flags() -> None:
    """Verify batch norm weights are frozen when batchnorm_onlybias is True."""
    m = Image2VectorCNN(img_size=(1, 32, 32), output_dim=3,
                        batchnorm_onlybias=True)
    # Check a representative BN in each submodule.
    assert m.interp_module.inNorm.weight.requires_grad is False
    assert m.reduction_module.inNorm.weight.requires_grad is False


def test_image2vector_reduction_sizes(small_image2vector_model: Image2VectorCNN
                                      ) -> None:
    """Ensure reduction yields spatial dims not exceeding the threshold.

    The reduction module computes finalH/finalW based on halving operations;
    those should be positive and less than or equal to the requested threshold.
    """
    h, w = small_image2vector_model.finalH, small_image2vector_model.finalW
    th_h, th_w = small_image2vector_model.size_threshold
    assert h > 0 and w > 0
    assert h <= th_h and w <= th_w


def test_image2vector_backward_pass(small_image2vector_model: Image2VectorCNN
                                    ) -> None:
    """Confirm gradients flow through the model on a simple loss."""
    m = small_image2vector_model.train()
    x = torch.randn(3, *m.img_size, requires_grad=False)
    y = m(x)
    loss = y.sum()
    loss.backward()
    # Pick a representative parameter and ensure it received gradients.
    assert m.endConv.weight.grad is not None


def test_image2vector_invalid_input_channels_raises(
    small_image2vector_model: Image2VectorCNN,
) -> None:
    """Passing a tensor with the wrong channel count should raise an error."""
    m = small_image2vector_model
    bad_x = torch.randn(1, 2, *m.img_size[1:])  # expecting 1 channel
    with pytest.raises(RuntimeError):
        _ = m(bad_x)
