"""This module contains the SupConClassifier implementation for MMClassification."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Optional

import torch
from mmpretrain.models.classifiers.image import ImageClassifier
from mmpretrain.registry import MODELS


@MODELS.register_module()
class SupConClassifier(ImageClassifier):
    """SupConClassifier with support for classification tasks."""

    def __init__(
        self,
        backbone: dict,
        neck: Optional[dict] = None,
        head: Optional[dict] = None,
        pretrained: Optional[dict] = None,
        **kwargs,
    ) -> None:
        self.multilabel = kwargs.pop("multilabel", False)
        self.hierarchical = kwargs.pop("hierarchical", False)
        super().__init__(backbone, neck=neck, head=head, pretrained=pretrained, **kwargs)

    def forward_train(self, img: torch.Tensor, gt_label: torch.Tensor, **kwargs) -> dict:
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
