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

    def forward(self, fc_feats, gt_labels=None, aux_feats=None):
        """
        Compute Barlow Twins Loss and, if labels are not none,
        also the Cross-Entropy loss.
        Args:
            fc_feats: tensor to train the linear classifier on
            labels: ground truth of shape [bsz].
            aux_feats: hidden vector of shape [bsz, n_views, ...].
        Returns:
            A dictionary containing the loss in the 'loss' key.
        """
        losses = dict()
        losses["loss"] = 0

        # Cross-Entropy loss: classification loss
        if fc_feats is not None and gt_labels is not None:
            gt_labels = gt_labels.squeeze(dim=1)
            if fc_feats.shape[0] == gt_labels.shape[0] * 2:
                losses["loss"] = self.criterion(
                    fc_feats, torch.cat([gt_labels, gt_labels], dim=0)
                )
            else:
                losses["loss"] = self.criterion(fc_feats, gt_labels)

        if aux_feats is None:
            losses["loss"] *= self.loss_weight
            return losses

        if len(aux_feats.shape) < 3:
            raise ValueError(
                "`aux_feats` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(aux_feats.shape) > 3:
            aux_feats = aux_feats.view(aux_feats.shape[0], aux_feats.shape[1], -1)

        batch_size = aux_feats.shape[0]
        dimensionality = aux_feats.shape[2]

        # Barlow Twins loss: redundancy reduction
        bn = nn.BatchNorm1d(dimensionality, affine=False, track_running_stats=False)
        # empirical cross-correlation matrix
        eccm = bn(aux_feats[:, 0, :]).T @ bn(aux_feats[:, 1, :])
        eccm.div_(batch_size)

        # Compute the invariance term (diagonal) and redundacy term (off-diagonal)
        on_diag = torch.diagonal(eccm).add(-1).pow_(2).sum()
        off_diag = off_diagonal(eccm).pow_(2).sum()
        # Normalize the loss by the dimensionality of the projector
        losses["loss"] += (on_diag + self.penalty * off_diag) / dimensionality

        losses["loss"] *= self.loss_weight
        return losses
