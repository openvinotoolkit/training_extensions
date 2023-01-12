# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
import torch.nn.functional as F

from mmcls.models.builder import HEADS, build_loss
from mmcls.models.heads.base_head import BaseHead


@HEADS.register_module()
class SupConMultiLabelClsHead(BaseHead):
    """
    Supervised Contrastive Learning head for Classification using SelfSL
    Args:
        num_classes (int): The number of classes of dataset used for training
        in_channels (int): The channels of input data from the backbone
        aux_mlp (dict): A dictionary with the out_channels and optionally the
                         hid_channels of the auxiliary MLP head.
        loss (dict): The classification loss (e.g. CrossEntropyLoss)
        aux_loss (dict): The SelfSL loss (e.g. BarlowTwinsLoss)
        scale (float): positive scale parameter.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        aux_mlp,
        loss=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
        aux_loss=dict(type="BarlowTwinsLoss", off_diag_penality=1.0 / 128.0, loss_weight=1.0),
        scale: float = 1.0,
        **kwargs,
    ):
        if in_channels <= 0:
            raise ValueError(f"in_channels={in_channels} must be a positive integer")
        if num_classes <= 0:
            raise ValueError("at least one class must be exist num_classes.")

        super().__init__()

        # Set up the losses
        self.compute_loss = build_loss(loss)
        self.aux_loss = build_loss(aux_loss)

        # Set up the standard classification head
        self.num_classes = num_classes
        self.classifier = nn.Linear(in_features=in_channels, out_features=self.num_classes)
        self.scale = scale

        # Set up the auxiliar head
        out_channels = aux_mlp["out_channels"]
        if out_channels <= 0:
            raise ValueError(f"out_channels={out_channels} must be a positive integer")
        if "hid_channels" in aux_mlp and aux_mlp["hid_channels"] > 0:
            hid_channels = aux_mlp["hid_channels"]
            self.aux_mlp = nn.Sequential(
                nn.Linear(in_features=in_channels, out_features=hid_channels),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=hid_channels, out_features=out_channels),
            )
        else:
            self.aux_mlp = nn.Linear(in_features=in_channels, out_features=out_channels)

    def loss(self, cls_score, gt_label, valid_label_mask=None):
        gt_label = gt_label.type_as(cls_score)
        num_samples = len(cls_score)

        # map difficult examples to positive ones
        _gt_label = torch.abs(gt_label)
        # compute loss
        loss = self.compute_loss(cls_score, _gt_label, valid_label_mask=valid_label_mask, avg_factor=num_samples)
        return loss / self.scale

    def forward_train(self, x, gt_label, **kwargs):
        img_metas = kwargs.get("img_metas", False)
        gt_label = gt_label.type_as(x)
        cls_score = self.classifier(x) * self.scale

        losses = dict(loss=0.0)

        bsz = gt_label.shape[0]
        # make sure we have two views for each label and split them
        assert x.shape[0] == 2 * bsz
        feats1, feats2 = torch.split(self.aux_mlp(x), [bsz, bsz], dim=0)
        gt_label = torch.cat([gt_label, gt_label], dim=0)

        if img_metas:
            num_samples = len(cls_score)
            gt_label = gt_label.type_as(cls_score)
            valid_label_mask = self.get_valid_label_mask(img_metas=img_metas)
            valid_label_mask = torch.cat([valid_label_mask, valid_label_mask], dim=0)
            loss = self.compute_loss(cls_score, gt_label, valid_label_mask=valid_label_mask, avg_factor=num_samples)
        else:
            loss = self.compute_loss(cls_score, gt_label)

        aux_loss = self.aux_loss(feats1, feats2)
        losses["loss"] = loss + aux_loss
        return losses

    def simple_test(self, img):
        """
        Test without data augmentation.
        """
        cls_score = self.classifier(img)

        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))

        pred = F.sigmoid(cls_score) if cls_score is not None else None

        if torch.onnx.is_in_onnx_export():
            return pred
        pred = list(pred.detach().cpu().numpy())
        return pred

    def get_valid_label_mask(self, img_metas):
        valid_label_mask = []
        for i, meta in enumerate(img_metas):
            mask = torch.Tensor([1 for _ in range(self.num_classes)])
            if "ignored_labels" in meta and meta["ignored_labels"]:
                mask[meta["ignored_labels"]] = 0
            mask = mask.cuda() if torch.cuda.is_available() else mask
            valid_label_mask.append(mask)
        valid_label_mask = torch.stack(valid_label_mask, dim=0)
        return valid_label_mask
