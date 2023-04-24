"""Cross entropy loss for ignored mode in class-incremental learning."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Optional

import torch
import torch.nn.functional as F
from mmseg.models.builder import LOSSES
from mmseg.models.losses import CrossEntropyLoss
from mmseg.models.losses.utils import weight_reduce_loss


@LOSSES.register_module()
class CrossEntropyLossWithIgnore(CrossEntropyLoss):
    """CrossEntropyLossWithIgnore with Ignore Mode Support for Class Incremental Learning."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._loss_name = "loss_ce_ignore"

    def forward(
        self,
        cls_score: Optional[torch.Tensor],
        label: Optional[torch.Tensor],
        weight: Optional[torch.Tensor] = None,
        avg_factor: Optional[int] = None,
        reduction_override: Optional[str] = "mean",
        ignore_index: int = 255,
        valid_label_mask: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """Forward."""
        if valid_label_mask is None:
            losses = super().forward(cls_score, label, weight, avg_factor, reduction_override, ignore_index, **kwargs)
            return losses
        else:
            batch_size = label.shape[0]
            for i in range(batch_size):
                invalid_labels = (valid_label_mask[i] == 0).nonzero(as_tuple=False)

                for inv_l in invalid_labels:
                    label[i] = torch.where(label[i] == inv_l.item(), ignore_index, label[i])

            losses = F.cross_entropy(cls_score, label, reduction="none", ignore_index=ignore_index)

            if weight is not None:
                weight = weight.float()
            losses = weight_reduce_loss(losses, weight=weight, reduction="mean", avg_factor=avg_factor)

            return losses

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
