"""This module contains the SupConClassifier implementation for MMClassification."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
from mmcls.models.builder import CLASSIFIERS
from mmcls.models.classifiers.image import ImageClassifier


@CLASSIFIERS.register_module()
class SupConClassifier(ImageClassifier):
    """SupConClassifier with support for classification tasks."""

    def __init__(self, backbone, neck=None, head=None, pretrained=None, task_adapt=None, **kwargs):
        self.multilabel = kwargs.pop("multilabel", False)
        self.hierarchical = kwargs.pop("hierarchical", False)
        self.task_adapt = task_adapt
        super().__init__(backbone, neck=neck, head=head, pretrained=pretrained, **kwargs)

    def forward_train(self, img, gt_label, **kwargs):
        """Concatenate the different image views along the batch size."""
        if len(img.shape) == 5:
            img = torch.cat([img[:, d, :, :, :] for d in range(img.shape[1])], dim=0)
        x = self.extract_feat(img)
        losses = dict()
        if self.multilabel or self.hierarchical:
            loss = self.head.forward_train(x, gt_label, **kwargs)
        else:
            gt_label = gt_label.squeeze(dim=1)
            loss = self.head.forward_train(x, gt_label)
        losses.update(loss)
        return losses
