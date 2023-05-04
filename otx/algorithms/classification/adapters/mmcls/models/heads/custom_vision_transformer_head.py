"""Module to define CustomVisionTransformerClsHead for classification task."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmcls.models.builder import HEADS
from mmcls.models.heads import VisionTransformerClsHead


@HEADS.register_module()
class CustomVisionTransformerClsHead(VisionTransformerClsHead):
    """Custom Vision Transformer classifier head which supports IBLoss loss calculation."""

    def loss(self, cls_score, gt_label, feature=None):
        """Calculate loss for given cls_score/gt_label."""
        num_samples = len(cls_score)
        losses = dict()
        # compute loss
        if self.compute_loss._get_name() == "IBLoss":
            loss = self.compute_loss(cls_score, gt_label, feature=feature)
        else:
            loss = self.compute_loss(cls_score, gt_label, avg_factor=num_samples)
        if self.cal_acc:
            # compute accuracy
            acc = self.compute_accuracy(cls_score, gt_label)
            assert len(acc) == len(self.topk)
            losses["accuracy"] = {f"top-{k}": a for k, a in zip(self.topk, acc)}
        losses["loss"] = loss
        return losses
