"""Module for defining non-linear classification head."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
import torch.nn.functional as F
from mmcls.models.builder import HEADS
from mmcls.models.heads.cls_head import ClsHead
from mmcv.cnn import build_activation_layer, constant_init, normal_init
from torch import nn


@HEADS.register_module()
class NonLinearClsHead(ClsHead):
    """Nonlinear classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        hid_channels (int): Number of channels of hidden layer.
        act_cfg (dict): Config of activation layer.
        loss (dict): Config of classification loss.
        topk (int | tuple): Top-k accuracy.
    """  # noqa: W605

    def __init__(
        self,
        num_classes,
        in_channels,
        hid_channels=1280,
        act_cfg=None,
        loss=None,
        topk=(1,),
        dropout=False,
        **kwargs,
    ):  # pylint: disable=too-many-arguments
        topk = (1,) if num_classes < 5 else (1, 5)
        act_cfg = act_cfg if act_cfg else dict(type="ReLU")
        loss = loss if loss else dict(type="CrossEntropyLoss", loss_weight=1.0)
        super().__init__(loss=loss, topk=topk, **kwargs)
        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.num_classes = num_classes
        self.act = build_activation_layer(act_cfg)
        self.dropout = dropout

        if self.num_classes <= 0:
            raise ValueError(f"num_classes={num_classes} must be a positive integer")

        self._init_layers()

    def _init_layers(self):
        if self.dropout:
            self.classifier = nn.Sequential(
                nn.Linear(self.in_channels, self.hid_channels),
                nn.BatchNorm1d(self.hid_channels),
                self.act,
                nn.Dropout(p=0.2),
                nn.Linear(self.hid_channels, self.num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.in_channels, self.hid_channels),
                nn.BatchNorm1d(self.hid_channels),
                self.act,
                nn.Linear(self.hid_channels, self.num_classes),
            )

    def init_weights(self):
        """Initialize weights of head."""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                normal_init(module, mean=0, std=0.01, bias=0)
            elif isinstance(module, nn.BatchNorm1d):
                constant_init(module, 1)

    def simple_test(self, img):
        """Test without augmentation."""
        cls_score = self.classifier(img)
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        if torch.onnx.is_in_onnx_export():
            return cls_score
        pred = F.softmax(cls_score, dim=1) if cls_score is not None else None
        pred = list(pred.detach().cpu().numpy())
        return pred

    def forward(self, x):
        """Forward fuction of NonLinearClsHead class."""
        return self.simple_test(x)

    def forward_train(self, cls_score, gt_label):
        """Forward_train fuction of NonLinearClsHead class."""
        logit = self.classifier(cls_score)
        losses = self.loss(logit, gt_label)
        return losses
