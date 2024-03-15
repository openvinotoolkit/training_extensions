"""Non-linear classification head for hierarhical classification task."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
from mmcls.models.builder import HEADS, build_loss
from mmcls.models.heads import MultiLabelClsHead
from mmcv.cnn import build_activation_layer, constant_init, normal_init
from torch import nn

from .mixin import OTXHeadMixin


@HEADS.register_module()
class CustomHierarchicalNonLinearClsHead(
    OTXHeadMixin, MultiLabelClsHead
):  # pylint: disable=too-many-instance-attributes
    """Custom NonLinear classification head for hierarchical classification task.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        hid_channels (int): Number of channels of hidden layer.
        act_cfg (dict): Config of activation layer.
        loss (dict): Config of classification loss.
        multilabel_loss (dict): Config of multi-label classification loss.
    """

    def __init__(
        self,
        num_classes,
        in_channels,
        hid_channels=1280,
        act_cfg=None,
        loss=None,
        multilabel_loss=None,
        dropout=False,
        **kwargs,
    ):  # pylint: disable=too-many-arguments
        act_cfg = act_cfg if act_cfg else dict(type="ReLU")
        loss = loss if loss else dict(type="CrossEntropyLoss", use_sigmoid=True, reduction="mean", loss_weight=1.0)
        multilabel_loss = (
            multilabel_loss if multilabel_loss else dict(type="AsymmetricLoss", reduction="mean", loss_weight=1.0)
        )
        self.hierarchical_info = kwargs.pop("hierarchical_info", None)
        assert self.hierarchical_info
        super().__init__(loss=loss)
        if self.hierarchical_info["num_multiclass_heads"] + self.hierarchical_info["num_multilabel_classes"] == 0:
            raise ValueError("Invalid classification heads configuration")
        self.compute_multilabel_loss = False
        if self.hierarchical_info["num_multilabel_classes"] > 0:
            self.compute_multilabel_loss = build_loss(multilabel_loss)

        if num_classes <= 0:
            raise ValueError(f"num_classes={num_classes} must be a positive integer")

        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.num_classes = num_classes
        self.act = build_activation_layer(act_cfg)
        self.dropout = dropout
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
        """Iniitialize weights of classification head."""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                normal_init(module, mean=0, std=0.01, bias=0)
            elif isinstance(module, nn.BatchNorm1d):
                constant_init(module, 1)

    def loss(self, cls_score, gt_label, multilabel=False, valid_label_mask=None):
        """Calculate loss for given cls_score and gt_label."""
        num_samples = len(cls_score)
        # compute loss
        if multilabel:
            gt_label = gt_label.type_as(cls_score)
            # map difficult examples to positive ones
            _gt_label = torch.abs(gt_label)

            loss = self.compute_multilabel_loss(
                cls_score, _gt_label, valid_label_mask=valid_label_mask, avg_factor=num_samples
            )
        else:
            loss = self.compute_loss(cls_score, gt_label, avg_factor=num_samples)

        return loss

    def forward(self, x):
        """Forward fuction of CustomHierarchicalNonLinearClsHead class."""
        return self.simple_test(x)

    def forward_train(self, cls_score, gt_label, **kwargs):
        """Forward_train fuction of CustomHierarchicalNonLinearClsHead class."""
        img_metas = kwargs.get("img_metas", None)
        cls_score = self.pre_logits(cls_score)
        gt_label = gt_label.type_as(cls_score)
        cls_score = self.classifier(cls_score)

        losses = dict(loss=0.0)
        num_effective_heads_in_batch = 0
        for i in range(self.hierarchical_info["num_multiclass_heads"]):
            if i not in self.hierarchical_info["empty_multiclass_head_indices"]:
                head_gt = gt_label[:, i]
                head_logits = cls_score[
                    :,
                    self.hierarchical_info["head_idx_to_logits_range"][str(i)][0] : self.hierarchical_info[
                        "head_idx_to_logits_range"
                    ][str(i)][1],
                ]
                valid_mask = head_gt >= 0
                head_gt = head_gt[valid_mask].long()
                if len(head_gt) > 0:
                    head_logits = head_logits[valid_mask, :]
                    multiclass_loss = self.loss(head_logits, head_gt)
                    losses["loss"] += multiclass_loss
                    num_effective_heads_in_batch += 1

        if num_effective_heads_in_batch > 0:
            losses["loss"] /= num_effective_heads_in_batch

        if self.compute_multilabel_loss:
            head_gt = gt_label[:, self.hierarchical_info["num_multiclass_heads"] :]
            head_logits = cls_score[:, self.hierarchical_info["num_single_label_classes"] :]
            valid_batch_mask = head_gt >= 0
            head_gt = head_gt[
                valid_batch_mask,
            ]
            head_logits = head_logits[
                valid_batch_mask,
            ]

            # multilabel_loss is assumed to perform no batch averaging
            if img_metas is not None:
                valid_label_mask = self.get_valid_label_mask(img_metas=img_metas)[
                    :, self.hierarchical_info["num_single_label_classes"] :
                ]
                valid_label_mask = valid_label_mask.to(cls_score.device)
                valid_label_mask = valid_label_mask[valid_batch_mask]
            else:
                valid_label_mask = None
            multilabel_loss = self.loss(head_logits, head_gt, multilabel=True, valid_label_mask=valid_label_mask)
            losses["loss"] += multilabel_loss.mean()
        return losses

    def simple_test(self, img):
        """Test without augmentation."""
        img = self.pre_logits(img)
        cls_score = self.classifier(img)
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))

        multiclass_logits = []
        for i in range(self.hierarchical_info["num_multiclass_heads"]):
            multiclass_logit = cls_score[
                :,
                self.hierarchical_info["head_idx_to_logits_range"][str(i)][0] : self.hierarchical_info[
                    "head_idx_to_logits_range"
                ][str(i)][1],
            ]
            if not torch.onnx.is_in_onnx_export():
                multiclass_logit = torch.softmax(multiclass_logit, dim=1)
            multiclass_logits.append(multiclass_logit)
        multiclass_pred = torch.cat(multiclass_logits, dim=1) if multiclass_logits else None

        if self.compute_multilabel_loss:
            multilabel_logits = cls_score[:, self.hierarchical_info["num_single_label_classes"] :]
            if not torch.onnx.is_in_onnx_export():
                multilabel_pred = torch.sigmoid(multilabel_logits) if multilabel_logits is not None else None
            else:
                multilabel_pred = multilabel_logits
            if multiclass_pred is not None:
                pred = torch.cat([multiclass_pred, multilabel_pred], axis=1)
            else:
                pred = multilabel_pred
        else:
            pred = multiclass_pred

        if torch.onnx.is_in_onnx_export():
            return pred
        pred = list(pred.detach().cpu().numpy())
        return pred

    def get_valid_label_mask(self, img_metas):
        """Get valid label mask with ignored_label."""
        valid_label_mask = []
        for meta in img_metas:
            mask = torch.Tensor([1 for _ in range(self.num_classes)])
            if "ignored_labels" in meta and meta["ignored_labels"]:
                mask[meta["ignored_labels"]] = 0
            valid_label_mask.append(mask)
        valid_label_mask = torch.stack(valid_label_mask, dim=0)
        return valid_label_mask
