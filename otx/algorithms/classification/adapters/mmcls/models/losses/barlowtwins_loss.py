"""Module for defining BarlowTwinsLoss for supcon in classification task."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from mmcls.models.builder import LOSSES
from torch import Tensor, nn


def off_diagonal(x: Tensor):
    """Return a tensor containing all the elements outside the diagonal of x."""
    assert x.shape[0] == x.shape[1]
    return x.flatten()[:-1].view(x.shape[0] - 1, x.shape[0] + 1)[:, 1:].flatten()


@LOSSES.register_module()
class BarlowTwinsLoss(nn.Module):
    """Barlow Twins Loss: https://arxiv.org/abs/2103.03230.

    Self-Supervised Learning via Redundancy Reduction
    Code adapted from https://github.com/facebookresearch/barlowtwins.
    """

    def __init__(self, off_diag_penality, loss_weight=1.0):
        super().__init__()
        self.penalty = off_diag_penality
        self.loss_weight = loss_weight

    def forward(self, feats1: Tensor, feats2: Tensor):
        """Compute Barlow Twins Loss and, if labels are not none, also the Cross-Entropy loss.

        Args:
            feats1 (torch.Tensor): vectors of shape [bsz, ...]. Corresponding to one of two views of the same samples.
            feats2 (torch.Tensor): vectors of shape [bsz, ...]. Corresponding to one of two views of the same samples.

        Returns:
            A floating point number describing the Barlow Twins loss
        """

        batch_size = feats1.shape[0]
        assert batch_size == feats2.shape[0]
        dimensionality = feats1.shape[1]
        assert dimensionality == feats2.shape[1]

        # Barlow Twins loss: redundancy reduction
        batch_norm = nn.BatchNorm1d(dimensionality, affine=False, track_running_stats=False)
        # empirical cross-correlation matrix
        eccm = batch_norm(feats1).T @ batch_norm(feats2)
        eccm.div_(batch_size)

        # Compute the invariance term (diagonal) and redundacy term (off-diagonal)
        on_diag = torch.diagonal(eccm).add(-1).pow_(2).sum()
        off_diag = off_diagonal(eccm).pow_(2).sum()
        # Normalize the loss by the dimensionality of the projector
        return self.loss_weight * (on_diag + self.penalty * off_diag) / dimensionality
