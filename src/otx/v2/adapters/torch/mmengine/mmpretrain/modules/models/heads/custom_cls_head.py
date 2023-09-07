"""Module defining for OTX Custom Non-linear classification head."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


import torch
from mmpretrain.models.builder import HEADS
from mmpretrain.models.heads import LinearClsHead

from .non_linear_cls_head import NonLinearClsHead


@HEADS.register_module()
class CustomNonLinearClsHead(NonLinearClsHead):
    """Custom Nonlinear classifier head."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_type = kwargs.get("loss", dict(type="CrossEntropyLoss"))["type"]

    def _get_loss(self, cls_score: torch.Tensor, data_samples, feature=None, **kwargs):
        num_samples = len(cls_score)
        losses = dict()
        if "gt_score" in data_samples[0]:
            # Batch augmentation may convert labels to one-hot format scores.
            gt_label = torch.stack([i.gt_score for i in data_samples])
        else:
            gt_label = torch.cat([i.gt_label for i in data_samples])
        # compute loss
        if self.loss_type == "IBLoss":
            loss = self.loss_module(cls_score, gt_label, feature=feature)
        else:
            loss = self.loss_module(cls_score, gt_label, avg_factor=num_samples)
        if self.cal_acc:
            # compute accuracy
            acc = self.compute_accuracy(cls_score, gt_label)
            assert len(acc) == len(self.topk)
            losses["accuracy"] = {f"top-{k}": a for k, a in zip(self.topk, acc)}
        losses["loss"] = loss
        return losses

    def loss(self, feats, data_samples, feature=None, **kwargs):
        """Calculate loss for given cls_score/gt_label."""
        cls_score = self.classifier(feats)
        losses = self._get_loss(cls_score, data_samples, feature, **kwargs)
        return losses

    def forward(self, feats):
        """Forward fuction of CustomNonLinearClsHead class."""
        return self.predict(feats)


@HEADS.register_module()
class CustomLinearClsHead(LinearClsHead):
    """Custom linear classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        init_cfg (dict | optional): The extra init config of layers.
            Defaults to use dict(type='Normal', layer='Linear', std=0.01).
    """

    def __init__(self, num_classes, in_channels, init_cfg=None, **kwargs):
        init_cfg = init_cfg if init_cfg else dict(type="Normal", layer="Linear", std=0.01)
        super().__init__(num_classes, in_channels, init_cfg=init_cfg, **kwargs)
        self.loss_type = kwargs.get("loss", dict(type="CrossEntropyLoss"))["type"]

    def _get_loss(self, cls_score: torch.Tensor, data_samples, feature=None, **kwargs):
        num_samples = len(cls_score)
        losses = dict()
        if "gt_score" in data_samples[0]:
            # Batch augmentation may convert labels to one-hot format scores.
            gt_label = torch.stack([i.gt_score for i in data_samples])
        else:
            gt_label = torch.cat([i.gt_label for i in data_samples])
        # compute loss
        if self.loss_type == "IBLoss":
            loss = self.loss_module(cls_score, gt_label, feature=feature)
        else:
            loss = self.loss_module(cls_score, gt_label, avg_factor=num_samples)
        if self.cal_acc:
            # compute accuracy
            acc = self.compute_accuracy(cls_score, gt_label)
            assert len(acc) == len(self.topk)
            losses["accuracy"] = {f"top-{k}": a for k, a in zip(self.topk, acc)}
        losses["loss"] = loss
        return losses

    def loss(self, feats, data_samples, **kwargs):
        """Calculate loss for given cls_score/gt_label."""
        cls_score = self.fc(feats)
        losses = self._get_loss(cls_score, data_samples, **kwargs)

        return losses

    def predict(self, feats, data_samples=None):
        """Test without augmentation."""
        cls_score = self.fc(feats)
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        if torch.onnx.is_in_onnx_export():
            return cls_score
        prediction = self._get_predictions(cls_score, data_samples)
        return prediction

    def forward(self, feats):
        """Forward fuction of CustomLinearHead class."""
        return self.predict(feats)
