"""Module for defining a semi-supervised multi-label classifier using mmcls."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmcls.models.builder import CLASSIFIERS

from otx.utils.logger import get_logger

from .custom_image_classifier import CustomImageClassifier

logger = get_logger()


@CLASSIFIERS.register_module()
class SemiSLMultilabelClassifier(CustomImageClassifier):
    """Semi-SL Multilabel Classifier which supports unlabeled data by overriding forward_train."""

    def forward_train(self, img, gt_label, **kwargs):
        """Data is transmitted as a classifier training function.

        Args:
            img (list[Tensor]): List of tensors of shape (1, C, H, W)
                Typically these should be mean centered and std scaled.
            gt_label (Tensor): Ground truth labels for the input labeled images
            kwargs (keyword arguments): Specific to concrete implementation
        """
        if "extra_0" not in kwargs:
            raise ValueError("'extra_0' does not exist in the dataset")
        if "img_strong" not in kwargs:
            raise ValueError("'img_strong' does not exist in the dataset")

        target = gt_label.squeeze()
        unlabeled_data = kwargs["extra_0"]
        x = {}
        x["labeled_weak"] = self.extract_feat(img)
        x["labeled_strong"] = self.extract_feat(kwargs["img_strong"])

        img_uw = unlabeled_data["img"]
        x["unlabeled_weak"] = self.extract_feat(img_uw)

        img_us = unlabeled_data["img_strong"]
        x["unlabeled_strong"] = self.extract_feat(img_us)

        losses = dict()
        loss = self.head.forward_train(x, target)
        losses.update(loss)

        return losses
