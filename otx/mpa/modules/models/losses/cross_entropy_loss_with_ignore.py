# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
import torch.nn.functional as F
from mmseg.models.builder import LOSSES
from mmseg.models.losses.utils import get_class_weight

from .mpa_pixel_base import MPABasePixelLoss


@LOSSES.register_module()
class CrossEntropyLossWithIgnore(MPABasePixelLoss):
    """CrossEntropyLossWithIgnore with Ignore Mode Support for Class Incremental Learning.

    Args:
        model_classes (list[str]): Model classes
        bg_aware (bool, optional): Whether to enable BG-aware loss
            'background' class would be added the start of model classes/label schema
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self, reduction="mean", loss_weight=None, **kwargs):
        super(CrossEntropyLossWithIgnore, self).__init__(**kwargs)

        self.reduction = reduction
        self.class_weight = get_class_weight(loss_weight)

    @property
    def name(self):
        return "ce_with_ignore"

    def _calculate(self, cls_score, label, valid_label_mask, scale):
        if cls_score.shape[0] == 0:
            return torch.tensor(0.0)

        batch_size = label.shape[0]
        label = torch.from_numpy(label).to(cls_score.device)
        probs_all = F.softmax(scale * cls_score, dim=1)
        losses_l = []
        for i in range(batch_size):
            probs_gathered = probs_all[i, valid_label_mask[i] == 1]
            probs_nomatch = probs_all[i, valid_label_mask[i] == 0]
            probs_gathered = torch.unsqueeze(probs_gathered, 0)
            probs_nomatch = torch.unsqueeze(probs_nomatch, 0)

            probs_gathered[:, 0] += probs_nomatch.sum(dim=1)
            each_prob_log = torch.log(probs_gathered)

            # X-entropy: NLL loss w/ log-probabilities & labels
            each_label = torch.unsqueeze(label[i], 0)
            each_label = each_label.to(cls_score.device)
            loss = F.nll_loss(each_prob_log, each_label, reduction="none", ignore_index=self.ignore_index)
            losses_l.append(loss)

        losses = torch.cat(losses_l, dim=0)

        return losses, cls_score
