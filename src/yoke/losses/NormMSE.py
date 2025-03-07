"""This loss function is a per-channel normalized version of mean squared error."""

import torch.nn as nn

class NormalizedMSELoss(nn.Module):
    def __init__(self, eps=1e-8, reduction='none'):
        super(NormalizedMSELoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred, target):
        target_mean = target.mean(dim=(0, 2, 3), keepdim=True)
        target_std = target.std(dim=(0, 2, 3), keepdim=True) + self.eps

        pred_norm = (pred - target_mean) / target_std
        target_norm = (target - target_mean) / target_std

        loss = (pred_norm - target_norm) ** 2
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss