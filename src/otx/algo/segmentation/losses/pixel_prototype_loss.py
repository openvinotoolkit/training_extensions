"""Pixel Prototype Cross Entropy Loss."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import absolute_import, division, print_function

import torch
import torch.nn.functional as F
from torch import nn

from otx.algo.segmentation.losses import CrossEntropyLossWithIgnore


class PPC(nn.Module):
    """Pixel-prototype contrastive loss."""

    def __init__(self, ignore_label=255):
        super(PPC, self).__init__()
        self.ignore_label = ignore_label

    def forward(self, contrast_logits, contrast_target):
        """Forward function."""
        loss_ppc = F.cross_entropy(contrast_logits, contrast_target.long(), ignore_index=self.ignore_label)
        return loss_ppc


class PPD(nn.Module):
    """Pixel-prototype distance loss."""

    def __init__(self, ignore_label=255):
        super().__init__()
        self.ignore_label = ignore_label

    def forward(self, contrast_logits, contrast_target):
        """Forward function."""
        contrast_logits = contrast_logits[contrast_target != self.ignore_label, :]
        contrast_target = contrast_target[contrast_target != self.ignore_label]

        logits = torch.gather(contrast_logits, 1, contrast_target[:, None].long())
        loss_ppd = (1 - logits).pow(2).mean()

        return loss_ppd


class PixelPrototypeCELoss(nn.Module):
    """Prototype based loss.

    Computes cross entropy loss beetwen computeted angles (prototypes and pixel embedings) and target.
    Includes pixel-prototype contrastive Learning and pixel-prototype distance optimization
    Args:
        loss_ppc_weight (float): weight for pixel-prototype contrastive loss. Default: 0.001
        loss_ppd_weight (float): weight for pixel-prototype distance loss. Default: 0.01
        ignore_index (int): index to ignore. Default: 255
        ignore_mode (bool): ignore mode, used for class incremental learning. Default: False
    """

    def __init__(self, loss_ppc_weight=0.01, loss_ppd_weight=0.001, ignore_index=255):
        super(PixelPrototypeCELoss, self).__init__()
        self._loss_name = "pixel_proto_ce_loss"
        ignore_index = ignore_index
        self.loss_ppc_weight = loss_ppc_weight
        self.loss_ppd_weight = loss_ppd_weight
        self.seg_criterion = CrossEntropyLossWithIgnore(ignore_index=ignore_index)

        self.ppc_criterion = PPC()
        self.ppd_criterion = PPD()

    def forward(self, seg_out, target, proto_logits, proto_targets, valid_label_mask):
        """Forward function."""
        if self.loss_ppc_weight > 0 and proto_logits is not None and proto_targets is not None:
            loss_ppc = self.ppc_criterion(proto_logits, proto_targets)
        else:
            loss_ppc = 0
        if self.loss_ppd_weight > 0 and proto_logits is not None and proto_targets is not None:
            loss_ppd = self.ppd_criterion(proto_logits, proto_targets)
        else:
            loss_ppd = 0

        loss = self.seg_criterion(seg_out, target.squeeze(1).long(), valid_label_mask=valid_label_mask)
        return loss + self.loss_ppc_weight * loss_ppc + self.loss_ppd_weight * loss_ppd
