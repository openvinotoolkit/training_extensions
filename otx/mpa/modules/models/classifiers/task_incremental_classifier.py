# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmcls.models.builder import CLASSIFIERS
from mmcls.models.classifiers.image import ImageClassifier


@CLASSIFIERS.register_module()
class TaskIncrementalLwF(ImageClassifier):
    def __init__(self, backbone, neck=None, head=None, pretrained=None):
        super(TaskIncrementalLwF, self).__init__(backbone, neck=neck, head=head, pretrained=pretrained)

    def forward_train(self, img, gt_label, **kwargs):
        soft_label = kwargs["soft_label"]
        x = self.extract_feat(img)
        losses = dict()
        loss = self.head.forward_train(x, gt_label, soft_label)
        losses.update(loss)

        return losses

    def extract_prob(self, img):
        """Test without augmentation."""
        x = self.extract_feat(img)
        return self.head.extract_prob(x), x
