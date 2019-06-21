import numpy as np
import torch
import torch.nn as nn

from .utils import weighted_loss
from ..registry import LOSSES


@weighted_loss
def balanced_l1_loss(pred,
                     target,
                     beta=1.0,
                     alpha=0.5,
                     gamma=1.5,
                     reduction='mean'):
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0

    diff = torch.abs(pred - target)
    b = np.e**(gamma / alpha) - 1
    loss = torch.where(
        diff < beta, alpha / b *
        (b * diff + 1) * torch.log(b * diff / beta + 1) - alpha * diff,
        gamma * diff + gamma / b - alpha * beta)

    return loss


@LOSSES.register_module
class BalancedL1Loss(nn.Module):
    """Balanced L1 Loss

    arXiv: https://arxiv.org/pdf/1904.02701.pdf (CVPR 2019)
    """

    def __init__(self,
                 alpha=0.5,
                 gamma=1.5,
                 beta=1.0,
                 reduction='mean',
                 loss_weight=1.0):
        super(BalancedL1Loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, **kwargs):
        loss_bbox = self.loss_weight * balanced_l1_loss(
            pred,
            target,
            weight,
            alpha=self.alpha,
            gamma=self.gamma,
            beta=self.beta,
            reduction=self.reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_bbox
