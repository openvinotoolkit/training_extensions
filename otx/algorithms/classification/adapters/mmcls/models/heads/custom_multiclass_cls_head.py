# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmcls.models.builder import HEADS
from mmcls.models.heads import LinearClsHead
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init, constant_init, build_activation_layer

from mmcls.models.builder import HEADS
from mmcls.models.heads.cls_head import ClsHead


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

    def __init__(self,
                 num_classes,
                 in_channels,
                 hid_channels=1280,
                 act_cfg=dict(type='ReLU'),
                 loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 topk=(1, ),
                 dropout=False):
        topk = (1, ) if num_classes < 5 else (1, 5)
        super(NonLinearClsHead, self).__init__(loss=loss, topk=topk)
        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.num_classes = num_classes
        self.act = build_activation_layer(act_cfg)
        self.dropout = dropout

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self._init_layers()

    def _init_layers(self):
        if self.dropout:
            self.classifier = nn.Sequential(
                nn.Linear(self.in_channels, self.hid_channels),
                nn.BatchNorm1d(self.hid_channels),
                self.act,
                nn.Dropout(p=0.2),
                nn.Linear(self.hid_channels, self.num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.in_channels, self.hid_channels),
                nn.BatchNorm1d(self.hid_channels),
                self.act,
                nn.Linear(self.hid_channels, self.num_classes)
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
        pred = F.softmax(cls_score, dim=1) if cls_score is not None else None
        if torch.onnx.is_in_onnx_export():
            return pred
        pred = list(pred.detach().cpu().numpy())
        return pred

    def forward_train(self, x, gt_label):
        cls_score = self.classifier(x)
        losses = self.loss(cls_score, gt_label)
        return losses

@HEADS.register_module()
class CustomNonLinearClsHead(NonLinearClsHead):
    """Custom Nonlinear classifier head.
    """
    def __init__(self, *args, **kwargs):
        super(CustomNonLinearClsHead, self).__init__(*args, **kwargs)
        self.loss_type = kwargs.get('loss', dict(type='CrossEntropyLoss'))['type']

    def loss(self, cls_score, gt_label, feature=None):
        num_samples = len(cls_score)
        losses = dict()
        # compute loss
        if self.loss_type == 'IBLoss':
            loss = self.compute_loss(cls_score, gt_label, feature=feature)
        else:
            loss = self.compute_loss(cls_score, gt_label, avg_factor=num_samples)
        if self.cal_acc:
            # compute accuracy
            acc = self.compute_accuracy(cls_score, gt_label)
            assert len(acc) == len(self.topk)
            losses['accuracy'] = {
                f'top-{k}': a
                for k, a in zip(self.topk, acc)
            }
        losses['loss'] = loss
        return losses

    def forward_train(self, x, gt_label):
        cls_score = self.classifier(x)
        losses = self.loss(cls_score, gt_label, feature=x)
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
    def __init__(self,
                 num_classes,
                 in_channels,
                 init_cfg=dict(type='Normal', layer='Linear', std=0.01),
                 *args,
                 **kwargs):
        self.loss_type = kwargs.get('loss', dict(type='CrossEntropyLoss'))['type']
        super(CustomLinearClsHead, self).__init__(num_classes,
                                                  in_channels,
                                                  init_cfg=init_cfg,
                                                  *args,
                                                  **kwargs)

    def loss(self, cls_score, gt_label, feature=None):
        num_samples = len(cls_score)
        losses = dict()
        # compute loss
        if self.loss_type == 'IBLoss':
            loss = self.compute_loss(cls_score, gt_label, feature=feature)
        else:
            loss = self.compute_loss(cls_score, gt_label, avg_factor=num_samples)
        if self.cal_acc:
            # compute accuracy
            acc = self.compute_accuracy(cls_score, gt_label)
            assert len(acc) == len(self.topk)
            losses['accuracy'] = {
                f'top-{k}': a
                for k, a in zip(self.topk, acc)
            }
        losses['loss'] = loss
        return losses

    def forward_train(self, x, gt_label):
        cls_score = self.fc(x)
        losses = self.loss(cls_score, gt_label, feature=x)
        return losses
