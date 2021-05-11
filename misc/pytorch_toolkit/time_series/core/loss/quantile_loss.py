import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles

    def loss(self, pred, gt, quantile):
        assert quantile > 0.0 and quantile < 1.0
        delta = gt - pred
        loss = quantile * F.relu(delta) + (1.0 - quantile) * F.relu(-delta)
        return loss.unsqueeze(1)

    def forward(self, pred, gt):
        loss = []
        for i, q in enumerate(self.quantiles):
            loss.append(
                self.loss(pred[:, :, i], gt[:, :, i], q)
            )
        loss = torch.mean(torch.sum(torch.cat(loss, axis=1), axis=1))
        return loss


class NormalizedQuantileLoss(nn.Module):
    def forward(self, pred, gt, quantile):
        assert quantile > 0.0 and quantile < 1.0
        delta = gt - pred
        weighted_errors = quantile * F.relu(delta) + (1.0 - quantile) * F.relu(-delta)
        quantile_loss = weighted_errors.mean()
        normaliser = gt.abs().mean() + 1e-9
        return 2 * quantile_loss / normaliser
