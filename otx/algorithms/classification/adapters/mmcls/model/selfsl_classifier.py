# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
from torch.nn.functional import softmax
from mmcls.models.builder import CLASSIFIERS
from mpa.modules.models.classifiers.sam_classifier import SAMImageClassifier


@CLASSIFIERS.register_module()
class SelfSLClassifier(SAMImageClassifier):
    def __init__(
        self,
        backbone=None,
        neck=None,
        head=None,
        pretrained=None,
        **kwargs
    ):
        super(SelfSLClassifier, self).__init__(
            backbone=backbone,
            neck=neck,
            head=head,
            pretrained=pretrained,
        )

    def forward_train(self, img, gt_label, **kwargs):
        img = [img[:, 0, :, :, :], img[:, 1, :, :, :]]
        img = torch.cat(img, dim=0)
        x = self.extract_feat(img)
        losses = dict()
        loss = self.head.forward_train(x, gt_label)
        losses.update(loss)
        return losses

    def extract_prob(self, img):
        """Test without augmentation."""
        x = self.extract_feat(img)
        return softmax(self.head.fc(x)), x
