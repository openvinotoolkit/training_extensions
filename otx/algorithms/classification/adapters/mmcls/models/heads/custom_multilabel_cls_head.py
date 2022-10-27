"""Implementation of model heads for multi-label classification."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcls.models.builder import HEADS
from mmcls.models.heads import MultiLabelClsHead
from mmcv.cnn import build_activation_layer, constant_init, normal_init


class AnglularLinear(nn.Module):
    """Computes cos of angles between input vectors and weights vectors.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output cosine logits.
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        self.weight.data.normal_().renorm_(2, 0, 1e-5).mul_(1e5)

    def forward(self, x):
        """Forward."""
        cos_theta = F.normalize(x.view(x.shape[0], -1), dim=1).mm(F.normalize(self.weight.t(), p=2, dim=0))
        return cos_theta.clamp(-1, 1)


@HEADS.register_module()
class CustomMultiLabelLinearClsHead(MultiLabelClsHead):
    """Custom Linear classification head for multilabel task.

    Args:
        num_classes (int): Number of categories.
        in_channels (int): Number of channels in the input feature map.
        normalized (bool): Normalize input features and weights.
        scale (float): positive scale parameter.
        loss (dict): Config of classification loss.
    """

    def __init__(
        self,
        num_classes,
        in_channels,
        normalized=False,
        scale=1.0,
        loss=dict(type="CrossEntropyLoss", use_sigmoid=True, reduction="mean", loss_weight=1.0),
    ):
        super(CustomMultiLabelLinearClsHead, self).__init__(loss=loss)
        if num_classes <= 0:
            raise ValueError(f"num_classes={num_classes} must be a positive integer")

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.normalized = normalized
        self.scale = scale
        self._init_layers()

    def _init_layers(self):
        if self.normalized:
            self.fc = AnglularLinear(self.in_channels, self.num_classes)
        else:
            self.fc = nn.Linear(self.in_channels, self.num_classes)

    def init_weights(self):
        """Initialize weights."""
        if isinstance(self.fc, nn.Linear):
            normal_init(self.fc, mean=0, std=0.01, bias=0)

    def loss(self, cls_score, gt_label, valid_label_mask=None):
        """Compute loss."""
        gt_label = gt_label.type_as(cls_score)
        num_samples = len(cls_score)
        losses = dict()

        # map difficult examples to positive ones
        _gt_label = torch.abs(gt_label)
        # compute loss
        loss = self.compute_loss(cls_score, _gt_label, valid_label_mask=valid_label_mask, avg_factor=num_samples)
        losses["loss"] = loss / self.scale
        return losses

    def forward_train(self, x, gt_label, **kwargs):
        """Forward."""
        img_metas = kwargs.get("img_metas", False)
        gt_label = gt_label.type_as(x)
        cls_score = self.fc(x) * self.scale
        if img_metas:
            valid_label_mask = self.get_valid_label_mask(img_metas=img_metas)
            losses = self.loss(cls_score, gt_label, valid_label_mask=valid_label_mask)
        else:
            losses = self.loss(cls_score, gt_label)
        return losses

    def simple_test(self, img):
        """Test without augmentation."""
        cls_score = self.fc(img) * self.scale
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        pred = torch.sigmoid(cls_score) if cls_score is not None else None
        if torch.onnx.is_in_onnx_export():
            return pred
        pred = list(pred.detach().cpu().numpy())
        return pred

    def get_valid_label_mask(self, img_metas):
        """Get valid label mask."""
        valid_label_mask = []
        for i, meta in enumerate(img_metas):
            mask = torch.Tensor([1 for _ in range(self.num_classes)])
            if "ignored_labels" in meta and meta["ignored_labels"]:
                mask[meta["ignored_labels"]] = 0
            mask = mask.cuda() if torch.cuda.is_available() else mask
            valid_label_mask.append(mask)
        valid_label_mask = torch.stack(valid_label_mask, dim=0)
        return valid_label_mask


@HEADS.register_module()
class CustomMultiLabelNonLinearClsHead(MultiLabelClsHead):
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

    def __init__(
        self,
        num_classes,
        in_channels,
        hid_channels=1280,
        act_cfg=dict(type="ReLU"),
        scale=1.0,
        loss=dict(type="CrossEntropyLoss", use_sigmoid=True, reduction="mean", loss_weight=1.0),
        dropout=False,
        normalized=False,
    ):

        super(CustomMultiLabelNonLinearClsHead, self).__init__(loss=loss)

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hid_channels = hid_channels
        self.act = build_activation_layer(act_cfg)
        self.dropout = dropout
        self.normalized = normalized
        self.scale = scale

        if self.num_classes <= 0:
            raise ValueError(f"num_classes={num_classes} must be a positive integer")

        self._init_layers()

    def _init_layers(self):
        modules = [nn.Linear(self.in_channels, self.hid_channels), nn.BatchNorm1d(self.hid_channels), self.act]
        if self.dropout:
            modules.append(nn.Dropout(p=0.2))
        if self.normalized:
            modules.append(AnglularLinear(self.hid_channels, self.num_classes))
        else:
            modules.append(nn.Linear(self.hid_channels, self.num_classes))

        self.classifier = nn.Sequential(*modules)

    def init_weights(self):
        """Initialize weights."""
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                normal_init(m, mean=0, std=0.01, bias=0)
            elif isinstance(m, nn.BatchNorm1d):
                constant_init(m, 1)

    def loss(self, cls_score, gt_label, valid_label_mask=None):
        """Compute loss."""
        gt_label = gt_label.type_as(cls_score)
        num_samples = len(cls_score)
        losses = dict()

        # map difficult examples to positive ones
        _gt_label = torch.abs(gt_label)
        # compute loss
        loss = self.compute_loss(cls_score, _gt_label, valid_label_mask=valid_label_mask, avg_factor=num_samples)
        losses["loss"] = loss / self.scale
        return losses

    def forward_train(self, x, gt_label, **kwargs):
        """Forward."""
        img_metas = kwargs.get("img_metas", False)
        gt_label = gt_label.type_as(x)
        cls_score = self.classifier(x) * self.scale
        if img_metas:
            valid_label_mask = self.get_valid_label_mask(img_metas=img_metas)
            losses = self.loss(cls_score, gt_label, valid_label_mask=valid_label_mask)
        else:
            losses = self.loss(cls_score, gt_label)
        return losses

    def simple_test(self, img):
        """Test without augmentation."""
        cls_score = self.classifier(img) * self.scale
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        pred = torch.sigmoid(cls_score) if cls_score is not None else None
        if torch.onnx.is_in_onnx_export():
            return pred
        pred = list(pred.detach().cpu().numpy())
        return pred

    def get_valid_label_mask(self, img_metas):
        """Get valid label mask."""
        valid_label_mask = []
        for i, meta in enumerate(img_metas):
            mask = torch.Tensor([1 for _ in range(self.num_classes)])
            if "ignored_labels" in meta and meta["ignored_labels"]:
                mask[meta["ignored_labels"]] = 0
            mask = mask.cuda() if torch.cuda.is_available() else mask
            valid_label_mask.append(mask)
        valid_label_mask = torch.stack(valid_label_mask, dim=0)
        return valid_label_mask
