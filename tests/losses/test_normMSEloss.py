import pytest
import torch

import pytest
import torch
from yoke.losses.NormMSE import NormalizedMSELoss

@pytest.fixture
def norm_mse():
    return NormalizedMSELoss()

def test_norm_mse_loss_zero(norm_mse):
    inp = torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]])
    target = torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]])
    loss = norm_mse(inp, target)
    assert loss.item() == 0.0

def test_norm_mse_loss_positive(norm_mse):
    inp = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    target = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    loss = norm_mse(inp, target)
    assert loss.item() == 0.0

def test_norm_mse_loss_non_zero(norm_mse):
    inp = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    target = torch.tensor([[[[4.0, 5.0], [6.0, 7.0]]]])
    loss = norm_mse(inp, target)
    expected_loss = torch.mean(((inp - target.mean(dim=(0, 2, 3), keepdim=True)) / (target.std(dim=(0, 2, 3), keepdim=True) + norm_mse.eps) - (target - target.mean(dim=(0, 2, 3), keepdim=True)) / (target.std(dim=(0, 2, 3), keepdim=True) + norm_mse.eps)) ** 2).item()
    assert loss.item() == expected_loss

def test_norm_mse_loss_negative(norm_mse):
    inp = torch.tensor([[[[-1.0, -2.0], [-3.0, -4.0]]]])
    target = torch.tensor([[[[-4.0, -5.0], [-6.0, -7.0]]]])
    loss = norm_mse(inp, target)
    expected_loss = torch.mean(((inp - target.mean(dim=(0, 2, 3), keepdim=True)) / (target.std(dim=(0, 2, 3), keepdim=True) + norm_mse.eps) - (target - target.mean(dim=(0, 2, 3), keepdim=True)) / (target.std(dim=(0, 2, 3), keepdim=True) + norm_mse.eps)) ** 2).item()
    assert loss.item() == expected_loss

def test_norm_mse_loss_mean_reduction():
    norm_mse = NormalizedMSELoss(reduction='mean')
    inp = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    target = torch.tensor([[[[4.0, 5.0], [6.0, 7.0]]]])
    loss = norm_mse(inp, target)
    expected_loss = torch.mean(((inp - target.mean(dim=(0, 2, 3), keepdim=True)) / (target.std(dim=(0, 2, 3), keepdim=True) + norm_mse.eps) - (target - target.mean(dim=(0, 2, 3), keepdim=True)) / (target.std(dim=(0, 2, 3), keepdim=True) + norm_mse.eps)) ** 2).mean().item()
    assert loss.item() == expected_loss

def test_norm_mse_loss_sum_reduction():
    norm_mse = NormalizedMSELoss(reduction='sum')
    inp = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    target = torch.tensor([[[[4.0, 5.0], [6.0, 7.0]]]])
    loss = norm_mse(inp, target)
    expected_loss = torch.mean(((inp - target.mean(dim=(0, 2, 3), keepdim=True)) / (target.std(dim=(0, 2, 3), keepdim=True) + norm_mse.eps) - (target - target.mean(dim=(0, 2, 3), keepdim=True)) / (target.std(dim=(0, 2, 3), keepdim=True) + norm_mse.eps)) ** 2).sum().item()
    assert loss.item() == expected_loss
