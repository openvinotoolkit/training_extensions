"""Module for defining a semi-supervised classifier using mmcls."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
from mmcls.models.builder import CLASSIFIERS

from otx.algorithms.common.utils.logger import get_logger

from .sam_classifier import SAMImageClassifier

logger = get_logger()


@CLASSIFIERS.register_module()
class SemiSLClassifier(SAMImageClassifier):
    """Semi-SL Classifier.

    This classifier supports unlabeled data by overriding forward_train
    """

    def forward_train(self, imgs, **kwargs):
        """Data is transmitted as a classifier training function.

        Args:
            imgs (list[Tensor]): List of tensors of shape (1, C, H, W)
                Typically these should be mean centered and std scaled.
            kwargs (keyword arguments): Specific to concrete implementation
        """
        if "gt_label" not in kwargs:
            raise ValueError("'gt_label' does not exist in the labeled image")
        if "extra_0" not in kwargs:
            raise ValueError("'extra_0' does not exist in the dataset")
        target = kwargs["gt_label"].squeeze(dim=1)
        unlabeled_data = kwargs["extra_0"]
        x = {}
        x["labeled"] = self.extract_feat(imgs)

        img_uw = unlabeled_data["img"]
        # weakly augmented images are used only for getting the pseudo label.
        # not required to calculate gradients.
        with torch.no_grad():
            x["unlabeled_weak"] = self.extract_feat(img_uw)

        img_us = unlabeled_data["img_strong"]
        x["unlabeled_strong"] = self.extract_feat(img_us)

        losses = dict()
        loss = self.head.forward_train(x, target)
        losses.update(loss)

        return losses
