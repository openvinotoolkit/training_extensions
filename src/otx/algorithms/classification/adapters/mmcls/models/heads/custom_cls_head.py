"""Module defining for OTX Custom Non-linear classification head."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
import torch.nn.functional as F
from mmcls.models.builder import HEADS
from mmcls.models.heads import LinearClsHead

from .non_linear_cls_head import NonLinearClsHead


@HEADS.register_module()
class CustomNonLinearClsHead(NonLinearClsHead):
    """Custom Nonlinear classifier head."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_type = kwargs.get("loss", dict(type="CrossEntropyLoss"))["type"]

    def loss(self, cls_score, gt_label, feature=None):
        """Calculate loss for given cls_score/gt_label."""
        num_samples = len(cls_score)
        losses = dict()
        # compute loss
        if self.loss_type == "IBLoss":
            loss = self.compute_loss(cls_score, gt_label, feature=feature)
        else:
            loss = self.compute_loss(cls_score, gt_label, avg_factor=num_samples)
        if self.cal_acc:
            # compute accuracy
            acc = self.compute_accuracy(cls_score, gt_label)
            assert len(acc) == len(self.topk)
            losses["accuracy"] = {f"top-{k}": a for k, a in zip(self.topk, acc)}
        losses["loss"] = loss
        return losses

    def forward(self, x):
        """Forward fuction of CustomNonLinearHead class."""
        return self.simple_test(x)

    def forward_train(self, cls_score, gt_label):
        """Forward_train fuction of CustomNonLinearHead class."""
        logit = self.classifier(cls_score)
        losses = self.loss(logit, gt_label, feature=cls_score)
        return losses


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

    def loss(self, cls_score, gt_label, feature=None):
        """Calculate loss for given cls_score/gt_label."""
        num_samples = len(cls_score)
        losses = dict()
        # compute loss
        if self.loss_type == "IBLoss":
            loss = self.compute_loss(cls_score, gt_label, feature=feature)
        else:
            loss = self.compute_loss(cls_score, gt_label, avg_factor=num_samples)
        if self.cal_acc:
            # compute accuracy
            acc = self.compute_accuracy(cls_score, gt_label)
            assert len(acc) == len(self.topk)
            losses["accuracy"] = {f"top-{k}": a for k, a in zip(self.topk, acc)}
        losses["loss"] = loss
        return losses

    def simple_test(self, img):
        """Test without augmentation."""
        cls_score = self.fc(img)
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        if torch.onnx.is_in_onnx_export():
            return cls_score
        pred = F.softmax(cls_score, dim=1) if cls_score is not None else None

        return self.post_process(pred)

    def forward(self, x):
        """Forward fuction of CustomLinearHead class."""
        return self.simple_test(x)

    def forward_train(self, x, gt_label):
        """Forward_train fuction of CustomLinearHead class."""
        cls_score = self.fc(x)
        losses = self.loss(cls_score, gt_label, feature=x)
        return losses
