"""Module for defining classification head for supcon."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
from mmcls.models.builder import HEADS, build_loss
from mmcls.models.heads.base_head import BaseHead
from torch import nn


@HEADS.register_module()
class SupConClsHead(BaseHead):
    """Supervised Contrastive Learning head for Classification using SelfSL.

    Args:
        num_classes (int): The number of classes of dataset used for training
        in_channels (int): The channels of input data from the backbone
        aux_mlp (dict): A dictionary with the out_channels and optionally the
                         hid_channels of the auxiliary MLP head.
        loss (dict): The SelfSL loss: BarlowTwinsLoss (default)
        topk (set): evaluation topk score, default is (1, )
    """

    def __init__(
        self, num_classes: int, in_channels: int, aux_mlp, loss, aux_loss, topk=(1,), init_cfg=None
    ):  # pylint: disable=too-many-arguments
        if in_channels <= 0:
            raise ValueError(f"in_channels={in_channels} must be a positive integer")
        if num_classes <= 0:
            raise ValueError("at least one class must be exist num_classes.")

        if isinstance(topk, int):
            topk = (topk,)
        for _topk in topk:
            assert _topk > 0, "Top-k should be larger than 0"
        topk = (1,) if num_classes < 5 else (1, 5)
        super().__init__(init_cfg=init_cfg)

        self.topk = topk
        self.compute_loss = build_loss(loss)
        self.aux_loss = build_loss(aux_loss)

        # Set up the standard classification head
        self.num_classes = num_classes
        self.fc = nn.Linear(in_features=in_channels, out_features=self.num_classes)

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

    def forward(self, x):
        """Forward fuction of SupConClsHead class."""
        return self.simple_test(x)

    def forward_train(self, x, gt_label):
        """Forward train head using the Supervised Contrastive Loss.

        Args:
            x (Tensor): features from the backbone.
            gt_label (Tensor): ground truth.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        losses = dict(loss=0.0)
        cls_score = self.fc(x)

        bsz = gt_label.shape[0]
        # make sure we have two views for each label and split them
        assert x.shape[0] == 2 * bsz
        feats1, feats2 = torch.split(self.aux_mlp(x), [bsz, bsz], dim=0)
        gt_label = torch.cat([gt_label, gt_label], dim=0)

        loss = self.compute_loss(cls_score, gt_label)
        aux_loss = self.aux_loss(feats1, feats2)
        losses["loss"] = loss + aux_loss
        return losses

    def simple_test(self, img):
        """Test without data augmentation."""
        cls_score = self.fc(img)
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        pred = F.softmax(cls_score, dim=1) if cls_score is not None else None
        if torch.onnx.is_in_onnx_export():
            return pred
        pred = list(pred.detach().cpu().numpy())
        return pred
