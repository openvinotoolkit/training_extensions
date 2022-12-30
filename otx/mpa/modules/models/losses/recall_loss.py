# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
import torch.nn.functional as F
from mmseg.models.builder import LOSSES
from mmseg.models.losses.pixel_base import BasePixelLoss
from mmseg.models.losses.utils import get_class_weight


def recallCE(input, target, class_weight=None, ignore_index=255):

    _, c, _, _ = input.size()

    pred = input.argmax(dim=1)
    idex = (pred != target).view(-1)

    # recall loss
    gt_counter = torch.ones((c)).to(target.device)
    gt_idx, gt_count = torch.unique(target, return_counts=True)

    gt_count[gt_idx == ignore_index] = gt_count[0].clone()
    gt_idx[gt_idx == ignore_index] = 0
    gt_counter[gt_idx] = gt_count.float()

    fn_counter = torch.ones((c)).to(target.device)
    fn = target.view(-1)[idex]
    fn_idx, fn_count = torch.unique(fn, return_counts=True)

    fn_count[fn_idx == ignore_index] = fn_count[0].clone()
    fn_idx[fn_idx == ignore_index] = 0
    fn_counter[fn_idx] = fn_count.float()

    if class_weight is not None:
        class_weight = 0.5 * (fn_counter / gt_counter) + 0.5 * class_weight
    else:
        class_weight = fn_counter / gt_counter

    loss = F.cross_entropy(input, target, weight=class_weight, reduction="none", ignore_index=ignore_index)

    return loss


def recallprecisionCE(input, target, fn_weight=None, fp_weight=None, ignore_index=255):

    raise NotImplementedError()


@LOSSES.register_module()
class RecallLoss(BasePixelLoss):
    """RecallLoss.
    https://arxiv.org/pdf/2106.14917.pdf

    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_mask (bool, optional): Whether to use mask cross entropy loss.
            Defaults to False.
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
    """

    def __init__(self, use_precision=False, class_weight=None, **kwargs):
        super(RecallLoss, self).__init__(**kwargs)

        self.use_precision = use_precision
        self.class_weight = get_class_weight(class_weight)

        if self.use_precision:
            self.cls_criterion = recallprecisionCE
        else:
            self.cls_criterion = recallCE

    @property
    def name(self):
        return "recall_loss"

    def _calculate(self, cls_score, label, scale, weight=None):
        class_weight = None
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)

        loss = self.cls_criterion(scale * cls_score, label, class_weight=class_weight, ignore_index=self.ignore_index)

        return loss, cls_score
