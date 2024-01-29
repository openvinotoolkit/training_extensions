# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module defining for OTX Custom Non-linear classification head."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
import torch.nn.functional as F
from mmpretrain.registry import MODELS
from mmpretrain.models.heads import ClsHead, LinearClsHead
from mmcv.cnn import build_activation_layer
from mmengine.model import constant_init, normal_init

if TYPE_CHECKING:
    from mmpretrain.structures import DataSample
    
class CustomNonLinearClsHead(ClsHead):
    def __init__(
        self,
        num_classes,
        in_channels,
        hid_channels=1280,
        act_cfg: dict = {
            "type": "HSwish"
        },
        loss: dict = {
            "type": "CrossEntropyLoss",
            "loss_weight": 1.0
        },
        dropout=False
    ):
        super().__init__(
            loss=loss,
            num_classes=num_classes,
            in_channels=in_channels
        )
        
        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.num_classes = num_classes
        self.act = build_activation_layer(act_cfg) 
        self.dropout = dropout
        self.loss_type = loss["type"]
        
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
    
    def loss(self, feats: tuple[torch.Tensor], data_samples: list[DataSample], **kwargs) -> dict:
        """Calculate losses from the classification score.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            data_samples (List[DataSample]): The annotation data of
                every samples.
            **kwargs: Other keyword arguments to forward the loss module.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        cls_score = self(feats)
        
        num_samples = len(cls_score)
        losses = {}
        
        if self.loss_type == "IBLoss":
            loss = super()._get_loss(cls_score, data_samples)
        
        return super()._get_loss(cls_score, data_samples, **kwargs)

@MODELS.register_module()
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
        bs = cls_score.shape[0]
        if bs == 1:
            cls_score = torch.cat([cls_score, cls_score], dim=0)
            gt_label = torch.cat([gt_label, gt_label], dim=0)
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
