# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcls.models.builder import HEADS
from mmcls.models.heads.cls_head import ClsHead
from mmcv.cnn import build_activation_layer, constant_init, normal_init


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
        act_cfg=dict(type="ReLU"),
        loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
        topk=(1,),
        dropout=False,
        *args,
        **kwargs,
    ):
        topk = (1,) if num_classes < 5 else (1, 5)
        super(NonLinearClsHead, self).__init__(loss=loss, topk=topk, *args, **kwargs)
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
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                normal_init(m, mean=0, std=0.01, bias=0)
            elif isinstance(m, nn.BatchNorm1d):
                constant_init(m, 1)

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

    def forward_train(self, x, gt_label):
        cls_score = self.classifier(x)
        losses = self.loss(cls_score, gt_label)
        return losses
