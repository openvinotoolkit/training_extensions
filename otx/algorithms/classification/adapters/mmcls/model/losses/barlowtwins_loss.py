# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
from mmcls.models.builder import LOSSES
from mmcls.models.losses import CrossEntropyLoss


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


@LOSSES.register_module()
class BarlowTwinsLoss(nn.Module):
    """
    Barlow Twins Loss: https://arxiv.org/abs/2103.03230.
    Self-Supervised Learning via Redundancy Reduction
    Code adapted from https://github.com/facebookresearch/barlowtwins.
    """

    def __init__(self, off_diag_penality, loss_weight=1.0):
        super(BarlowTwinsLoss, self).__init__()
        self.penalty = off_diag_penality
        self.loss_weight = loss_weight
        self.criterion = CrossEntropyLoss()

    def forward(self, features, labels=None, fc_feats=None):
        """
        Compute Barlow Twins Loss and, if labels are not none,
        also the Cross-Entropy loss.
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            fc_feats: tensor to train the linear classifier on
        Returns:
            A dictionary containing the loss in the 'loss' key.
        """
        losses = dict()
        losses["loss"] = 0

        # Cross-Entropy loss: classification loss
        if fc_feats is not None and labels is not None:
            labels = labels.squeeze(dim=1)
            if fc_feats.shape[0] == labels.shape[0] * 2:
                losses["loss"] = self.criterion(
                    fc_feats, torch.cat([labels, labels], dim=0)
                )
            else:
                losses["loss"] = self.criterion(fc_feats, labels)

        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        dimensionality = features.shape[2]

        # Barlow Twins loss: redundancy reduction
        bn = nn.BatchNorm1d(dimensionality, affine=False, track_running_stats=False)
        # empirical cross-correlation matrix
        eccm = bn(features[:, 0, :]).T @ bn(features[:, 1, :])
        eccm.div_(batch_size)

        # Compute the invariance term (diagonal) and redundacy term (off-diagonal)
        on_diag = torch.diagonal(eccm).add(-1).pow_(2).sum()
        off_diag = off_diagonal(eccm).pow_(2).sum()
        # Normalize the loss by the dimensionality of the projector
        losses["loss"] += (on_diag + self.penalty * off_diag) / dimensionality

        losses["loss"] *= self.loss_weight
        return losses
