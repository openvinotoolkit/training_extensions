# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
import torch.nn as nn
from mmcls.models.builder import CLASSIFIERS, build_backbone, build_head, build_neck

from otx.mpa.modules.models.classifiers.sam_classifier import SAMClassifier
from otx.mpa.utils.logger import get_logger

logger = get_logger()


@CLASSIFIERS.register_module()
class SemiSLClassifier(SAMClassifier):
    """Semi-SL Classifier

    The classifier is a classifier that supports Semi-SL task
    that handles unlabeled data.

    Args:
        backbone (dict): backbone network configuration
        neck (dict): model neck configuration
        head (dict): model head configuration
        pretrained (str or boolean): Initialize to pre-trained weight
            according to the backbone when the path
            or boolean True of pre-trained weight is performed.
    """

    def __init__(self, backbone, neck=None, head=None, pretrained=None, **kwargs):
        super(SemiSLClassifier, self).__init__()
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        if head is not None:
            self.head = build_head(head)
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        """Initialize weight of backbone, neck, head
        pretrained arg for using the pretrained weight of the backbone

        Args:
            pretrained (str or boolean): pre-trained weight of the backbone
        """
        super(SemiSLClassifier, self).init_weights(pretrained)
        if pretrained is not None:
            logger.info("pretrained model: {}".format(pretrained))
        self.backbone.init_weights(pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_head:
            self.head.init_weights()

    def extract_feat(self, imgs):
        """Directly extract features from the backbone + neck

        Args:
            imgs (list[Tensor]): List of tensors of shape (1, C, H, W)
        """
        x = self.backbone(imgs)

        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self, imgs, **kwargs):
        """Data is transmitted as a classifier training function

        Args:
            imgs (list[Tensor]): List of tensors of shape (1, C, H, W)
                Typically these should be mean centered and std scaled.
            kwargs (keyword arguments): Specific to concrete implementation
        """
        if "gt_label" not in kwargs:
            raise ValueError("'gt_label' does not exist in the labeled image")
        if "extra_0" not in kwargs:
            raise ValueError("'extra_0' does not exist in the dataset")
        target = kwargs["gt_label"]
        unlabeled_data = kwargs["extra_0"]

        x = {}
        x["labeled"] = self.extract_feat(imgs)

        img_uw = unlabeled_data["weak"]["img"]
        # weakly augmented images are used only for getting the pseudo label.
        # not required to calculate gradients.
        with torch.no_grad():
            x["unlabeled_weak"] = self.extract_feat(img_uw)

        img_us = unlabeled_data["strong"]["img"]
        x["unlabeled_strong"] = self.extract_feat(img_us)

        losses = dict()
        loss = self.head.forward_train(x, target)
        losses.update(loss)

        return losses

    def simple_test(self, img, **kwargs):
        x = self.extract_feat(img)
        return self.head.simple_test(x)
