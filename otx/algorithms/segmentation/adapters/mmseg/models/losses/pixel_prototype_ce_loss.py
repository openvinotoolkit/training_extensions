from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F

class PPC(nn.Module, ABC):
    def __init__(self):
        super(PPC, self).__init__()

    def forward(self, contrast_logits, contrast_target):
        loss_ppc = F.cross_entropy(contrast_logits, contrast_target.long(), ignore_index=255)
        return loss_ppc


class PPD(nn.Module, ABC):
    def __init__(self):
        super(PPD, self).__init__()
        self.ignore_label = 255

    def forward(self, contrast_logits, contrast_target):
        contrast_logits = contrast_logits[contrast_target != self.ignore_label, :]
        contrast_target = contrast_target[contrast_target != self.ignore_label]

        logits = torch.gather(contrast_logits, 1, contrast_target[:, None].long())
        loss_ppd = (1 - logits).pow(2).mean()

        return loss_ppd

class PixelPrototypeCELoss(nn.Module, ABC):
    def __init__(self, loss_ppc_weight=0.01, loss_ppd_weight=0.001):
        super(PixelPrototypeCELoss, self).__init__()

        ignore_index = 255
        self.loss_ppc_weight = loss_ppc_weight
        self.loss_ppd_weight = loss_ppd_weight

        self.seg_criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

        self.ppc_criterion = PPC()
        self.ppd_criterion = PPD()

    def forward(self, preds, target):
        assert "seg_out" in preds
        assert "proto_logits" in preds
        assert "proto_targets" in preds

        pred = preds['seg_out']
        contrast_logits = preds['proto_logits']
        contrast_target = preds['proto_targets']
        loss_ppc = self.ppc_criterion(contrast_logits, contrast_target)
        loss_ppd = self.ppd_criterion(contrast_logits, contrast_target)
        logits, _ = torch.max(pred, dim=1)
        loss = self.seg_criterion(logits, target.squeeze(1))
        return loss + self.loss_ppc_weight * loss_ppc + self.loss_ppd_weight * loss_ppd
