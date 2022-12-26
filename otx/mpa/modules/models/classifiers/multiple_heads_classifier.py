# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch.nn as nn

from mmcls.models.builder import CLASSIFIERS, build_head
from mmcls.models.classifiers.image import ImageClassifier


@CLASSIFIERS.register_module()
class MultipleHeadsClassifier(ImageClassifier):
    def __init__(self, backbone, neck=None, heads=None, pretrained=None):
        if heads is not None:
            assert isinstance(heads, list)
        else:
            heads = [None]

        super(MultipleHeadsClassifier, self).__init__(
            backbone, neck=neck, head=heads[0], pretrained=pretrained
        )
        self.heads = nn.ModuleList()
        self.heads.append(self.head)
        delattr(self, "head")
        for head in heads[1:]:
            self.heads.append(build_head(head))

    @property
    def with_heads(self):
        return hasattr(self, "heads") and self.heads is not None

    def forward_train(self, img, gt_label, **kwargs):
        assert len(gt_label) == len(self.heads)

        if self.augments is not None:
            img, gt_label = self.augments(img, gt_label)

        x = self.extract_feat(img)

        losses = dict()
        for i, head in enumerate(self.heads):
            loss = head.forward_train(x, gt_label[i])
            loss_ = dict()
            for key, value in loss.items():
                loss_[f"{i}_{key}"] = value
            losses.update(loss_)
        return losses

    def simple_test(self, img, img_metas):
        """Test without augmentation."""
        x = self.extract_feat(img)
        x_dims = len(x.shape)
        if x_dims == 1:
            x.unsqueeze_(0)
        outs = []
        for head in self.heads:
            outs.append(head.simple_test(x))
        return outs
