# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
from mmcls.models.builder import CLASSIFIERS

from otx.mpa.utils.logger import get_logger

from .sam_classifier import SAMImageClassifier

logger = get_logger()


@CLASSIFIERS.register_module()
class SemiSLMultilabelClassifier(SAMImageClassifier):
    """Semi-SL Multilabel Classifier
    This classifier supports unlabeled data by overriding forward_train
    """

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
        if "img_strong" not in kwargs:
            raise ValueError("'img_strong' does not exist in the dataset")

        target = kwargs["gt_label"].squeeze()
        unlabeled_data = kwargs["extra_0"]
        x = {}
        x["labeled_weak"] = self.extract_feat(imgs)
        x["labeled_strong"] = self.extract_feat(kwargs["img_strong"])

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
