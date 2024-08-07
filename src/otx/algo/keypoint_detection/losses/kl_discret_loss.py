# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Implementation of KLDiscretLoss."""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional


class KLDiscretLoss(nn.Module):
    """Discrete KL Divergence loss for SimCC with Gaussian Label Smoothing.

    Modified from `the official implementation.

    <https://github.com/leeyegy/SimCC>`_.

    Args:
        beta (float): Temperature factor of Softmax. Default: 1.0.
        label_softmax (bool): Whether to use Softmax on labels.
            Default: False.
        label_beta (float): Temperature factor of Softmax on labels.
            Default: 10.0.
        use_target_weight (bool): Option to use weighted loss.
            Different joint types may have different target weights.
        mask (Tensor): Index of masked keypoints.
        mask_weight (float): Weight of masked keypoints. Default: 1.0.
    """

    def __init__(
        self,
        beta: float = 1.0,
        label_softmax: bool = False,
        label_beta: float = 10.0,
        use_target_weight: bool = True,
        mask: Tensor | None = None,
        mask_weight: float = 1.0,
    ):
        super().__init__()
        self.beta = beta
        self.label_softmax = label_softmax
        self.label_beta = label_beta
        self.use_target_weight = use_target_weight
        self.mask = mask
        self.mask_weight = mask_weight

        self.log_softmax = nn.LogSoftmax(dim=1)
        self.kl_loss = nn.KLDivLoss(reduction="none")

    def criterion(self, dec_outs: Tensor, labels: Tensor) -> Tensor:
        """Criterion function."""
        log_pt = self.log_softmax(dec_outs * self.beta)
        if self.label_softmax:
            labels = functional.softmax(labels * self.label_beta, dim=1)
        return torch.mean(self.kl_loss(log_pt, labels), dim=1)

    def forward(
        self,
        pred_simcc: tuple[Tensor, Tensor],
        gt_simcc: tuple[Tensor, Tensor],
        target_weight: Tensor,
    ) -> Tensor:
        """Forward function.

        Args:
            pred_simcc (Tuple[Tensor, Tensor]): Predicted SimCC vectors of
                x-axis and y-axis.
            gt_simcc (Tuple[Tensor, Tensor]): Target representations.
            target_weight (torch.Tensor[N, K] or torch.Tensor[N]):
                Weights across different labels.

        Returns:
            loss (Tensor): KL discrete loss between pred_simcc and gt_simcc.
        """
        batch_size, num_keypoints, _ = pred_simcc[0].shape
        loss = 0
        weight = target_weight.reshape(-1) if self.use_target_weight else 1.0

        for pred, target in zip(pred_simcc, gt_simcc):
            _pred = pred.reshape(-1, pred.size(-1))
            _target = target.reshape(-1, target.size(-1))

            t_loss = self.criterion(_pred, _target).mul(weight)

            if self.mask is not None:
                t_loss = t_loss.reshape(batch_size, num_keypoints)
                t_loss[:, self.mask] = t_loss[:, self.mask] * self.mask_weight

            loss = loss + t_loss.sum()

        return loss / num_keypoints
