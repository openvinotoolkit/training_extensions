from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.builder import LOSSES
from otx.algorithms.segmentation.adapters.mmseg.models.losses import CrossEntropyLossWithIgnore

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


@LOSSES.register_module()
class PixelPrototypeCELoss(nn.Module, ABC):
    def __init__(self,
                 loss_ppc_weight=0.01,
                 loss_ppd_weight=0.001,
                 ignore_index=255,
                 ignore_mode=True,
                 **kwargs):
        super(PixelPrototypeCELoss, self).__init__()
        self._loss_name = 'pixel_proto_ce_loss'
        ignore_index = ignore_index
        self.loss_ppc_weight = loss_ppc_weight
        self.loss_ppd_weight = loss_ppd_weight
        if not ignore_mode:
            self.seg_criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        else:
            self.seg_criterion = CrossEntropyLossWithIgnore(**kwargs)

        self.ppc_criterion = PPC()
        self.ppd_criterion = PPD()

    def forward(self, seg_out, proto_logits, proto_targets, target):
        if  self.loss_ppc_weight > 0:
            loss_ppc = self.ppc_criterion(proto_logits, proto_targets)
        else:
            loss_ppc = 0
        if  self.loss_ppd_weight > 0:
            loss_ppd = self.ppd_criterion(proto_logits, proto_targets)
        else:
            loss_ppd = 0
        loss = self.seg_criterion(seg_out, target.squeeze(1).long())
        return loss + self.loss_ppc_weight * loss_ppc + self.loss_ppd_weight * loss_ppd

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
