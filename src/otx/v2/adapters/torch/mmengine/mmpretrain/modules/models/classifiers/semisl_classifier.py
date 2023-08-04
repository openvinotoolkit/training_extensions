"""Module for defining a semi-supervised classifier using mmpretrain."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
from mmpretrain.models import ClsDataPreprocessor
from mmpretrain.registry import MODELS
from otx.v2.api.utils.logger import get_logger

from .custom_image_classifier import CustomImageClassifier

logger = get_logger()


@MODELS.register_module()
class SemiSLClsDataPreprocessor(ClsDataPreprocessor):
    def forward(self, data: dict, training: bool = False) -> dict:
        extra_0 = data.get("extra_0", None)
        inputs = super().forward(data, training)
        # Unlabeled Data forward to ClsDataPreprocessor
        if extra_0 is not None:
            # Unlabeled data with weak augmentation
            extra_strong = {"inputs": extra_0["img_strong"]}
            extra_0 = super().forward(extra_0, training)
            # Unlabeled data with strong augmentation
            extra_strong = super().forward(extra_strong, training)
            extra_0["img_strong"] = extra_strong
            inputs["extra_0"] = extra_0
        return inputs


@MODELS.register_module()
class SemiSLClassifier(CustomImageClassifier):
    """Semi-SL Classifier.

    This classifier supports unlabeled data by overriding forward_train
    """

    def __init__(self, **kwargs):
        data_preprocessor = kwargs.pop("data_preprocessor", {})
        train_cfg = kwargs.get("train_cfg", None)
        if data_preprocessor is None:
            data_preprocessor = {}
        if isinstance(data_preprocessor, dict):
            data_preprocessor.setdefault("type", "SemiSLClsDataPreprocessor")
            data_preprocessor.setdefault("batch_augments", train_cfg)
            data_preprocessor = MODELS.build(data_preprocessor)
        super().__init__(data_preprocessor=data_preprocessor, **kwargs)

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
        target = kwargs["gt_label"]
        unlabeled_data = kwargs["extra_0"]
        x = {}
        x["labeled"] = self.extract_feat(imgs)

        img_uw = unlabeled_data["inputs"]
        # weakly augmented images are used only for getting the pseudo label.
        # not required to calculate gradients.
        with torch.no_grad():
            x["unlabeled_weak"] = self.extract_feat(img_uw)

        img_us = unlabeled_data["img_strong"]["inputs"]
        x["unlabeled_strong"] = self.extract_feat(img_us)

        losses = dict()
        loss = self.head.forward_train(x, target)
        losses.update(loss)

        return losses

    def loss(self, inputs, data_samples, **kwargs):
        if "gt_score" in data_samples[0]:
            # Batch augmentation may convert labels to one-hot format scores.
            gt_label = torch.stack([i.gt_score for i in data_samples])
        else:
            gt_label = torch.cat([i.gt_label for i in data_samples])
        return self.forward_train(imgs=inputs, gt_label=gt_label, **kwargs)

    def forward(self, inputs, data_samples=None, extra_0=None, mode: str = "tensor"):
        if mode == "tensor":
            feats = self.extract_feat(inputs)
            return self.head(feats) if self.with_head else feats
        elif mode == "loss":
            return self.loss(inputs, data_samples, extra_0=extra_0)
        elif mode == "predict":
            return self.predict(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')
