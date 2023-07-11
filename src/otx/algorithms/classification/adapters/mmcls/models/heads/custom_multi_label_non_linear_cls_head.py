"""This module contains the CustomMultiLabelNonLinearClsHead implementation for MMClassification."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
from mmcls.models.builder import HEADS
from mmcls.models.heads import MultiLabelClsHead
from mmcv.cnn import build_activation_layer, constant_init, normal_init
from torch import nn

from .custom_multi_label_linear_cls_head import AnglularLinear
from .mixin import OTXHeadMixin


@HEADS.register_module()
class CustomMultiLabelNonLinearClsHead(OTXHeadMixin, MultiLabelClsHead):
    """Non-linear classification head for multilabel task.

    Args:
        num_classes (int): Number of categories.
        in_channels (int): Number of channels in the input feature map.
        loss (dict): Config of classification loss.
        scale (float): positive scale parameter.
        init_cfg (dict | optional): The extra init config of layers.
            Defaults to use dict(type='Normal', layer='Linear', std=0.01).
        normalized (bool): Normalize input features and weights in the last linar layer.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        num_classes,
        in_channels,
        hid_channels=1280,
        act_cfg=None,
        scale=1.0,
        loss=None,
        dropout=False,
        normalized=False,
    ):
        act_cfg = act_cfg if act_cfg else dict(type="ReLU")
        loss = loss if loss else dict(type="CrossEntropyLoss", use_sigmoid=True, reduction="mean", loss_weight=1.0)
        super().__init__(loss=loss)

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hid_channels = hid_channels
        self.dropout = dropout
        self.normalized = normalized
        self.scale = scale

        if self.num_classes <= 0:
            raise ValueError(f"num_classes={num_classes} must be a positive integer")

        self._init_layers(act_cfg)

    def _init_layers(self, act_cfg):
        modules = [
            nn.Linear(self.in_channels, self.hid_channels),
            nn.BatchNorm1d(self.hid_channels),
            build_activation_layer(act_cfg),
        ]
        if self.dropout:
            modules.append(nn.Dropout(p=0.2))
        if self.normalized:
            modules.append(AnglularLinear(self.hid_channels, self.num_classes))
        else:
            modules.append(nn.Linear(self.hid_channels, self.num_classes))

        self.classifier = nn.Sequential(*modules)

    def init_weights(self):
        """Iniitalize weights of model."""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                normal_init(module, mean=0, std=0.01, bias=0)
            elif isinstance(module, nn.BatchNorm1d):
                constant_init(module, 1)

    def loss(self, cls_score, gt_label, valid_label_mask=None):
        """Calculate loss for given cls_score/gt_label."""
        gt_label = gt_label.type_as(cls_score)
        num_samples = len(cls_score)
        losses = dict()

        # map difficult examples to positive ones
        _gt_label = torch.abs(gt_label)
        # compute loss
        loss = self.compute_loss(
            cls_score,
            _gt_label,
            valid_label_mask=valid_label_mask,
            avg_factor=num_samples,
        )
        losses["loss"] = loss / self.scale
        return losses

    def forward(self, x):
        """Forward fuction of CustomMultiLabelNonLinearClsHead."""
        return self.simple_test(x)

    def forward_train(self, cls_score, gt_label, **kwargs):
        """Forward_train fuction of CustomMultiLabelNonLinearClsHead."""
        img_metas = kwargs.get("img_metas", False)
        cls_score = self.pre_logits(cls_score)
        gt_label = gt_label.type_as(cls_score)
        cls_score = self.classifier(cls_score) * self.scale

        valid_batch_mask = gt_label >= 0
        gt_label = gt_label[
            valid_batch_mask,
        ].view(gt_label.shape[0], -1)
        cls_score = cls_score[
            valid_batch_mask,
        ].view(cls_score.shape[0], -1)
        if img_metas:
            valid_label_mask = self.get_valid_label_mask(img_metas=img_metas)
            valid_label_mask = valid_label_mask.to(cls_score.device)
            valid_label_mask = valid_label_mask[
                valid_batch_mask,
            ].view(valid_label_mask.shape[0], -1)
            losses = self.loss(cls_score, gt_label, valid_label_mask=valid_label_mask)
        else:
            losses = self.loss(cls_score, gt_label)
        return losses

    def simple_test(self, img):
        """Test without augmentation."""
        img = self.pre_logits(img)
        cls_score = self.classifier(img) * self.scale
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        if torch.onnx.is_in_onnx_export():
            return cls_score
        pred = torch.sigmoid(cls_score) if cls_score is not None else None
        pred = list(pred.detach().cpu().numpy())
        return pred

    def get_valid_label_mask(self, img_metas):
        """Get valid label with ignored_label mask."""
        valid_label_mask = []
        for meta in img_metas:
            mask = torch.Tensor([1 for _ in range(self.num_classes)])
            if "ignored_labels" in meta and meta["ignored_labels"]:
                mask[meta["ignored_labels"]] = 0
            valid_label_mask.append(mask)
        valid_label_mask = torch.stack(valid_label_mask, dim=0)
        return valid_label_mask
