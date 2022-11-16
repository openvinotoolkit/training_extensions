# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict

from mmcls.models.builder import HEADS, build_loss
from mmcls.models.heads.base_head import BaseHead


@HEADS.register_module()
class SelfSLClsHead(BaseHead):
    """
    SelfSL head for Classification Loss
    Args:
        num_classes (int): The number of classes of dataset used for training
        in_channels (int): The channels of input data from the backbone
        aux_head (dict): A dictionary with the out_channels and optionally the
                         hid_channels of the auxiliary head.
        loss (dict): The SelfSL loss: BarlowTwinsLoss (default)
        topk (set): evaluation topk score, default is (1, )
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        aux_head: Dict,
        loss=dict(type="BarlowTwinsLoss", off_diag_penality=1 / 128),
        lamda=1.0,
        topk=(1,),
        init_cfg=None,
    ):
        if in_channels <= 0:
            raise ValueError(f"in_channels={in_channels} must be a positive integer")
        if num_classes <= 0:
            raise ValueError("at least one class must be exist num_classes.")

        if isinstance(topk, int):
            topk = (topk,)
        for _topk in topk:
            assert _topk > 0, "Top-k should be larger than 0"
        super(BaseHead, self).__init__(init_cfg=init_cfg)

        self.topk = topk
        self.compute_loss = build_loss(loss)
        self.lamda = lamda

        # Set up the standard classification head
        self.num_classes = num_classes
        self.fc = nn.Linear(in_features=in_channels, out_features=self.num_classes)

        # Set up the auxiliar head
        out_channels = aux_head["out_channels"]
        if out_channels <= 0:
            raise ValueError(f"out_channels={out_channels} must be a positive integer")
        if "hid_channels" in aux_head and aux_head["hid_channels"] > 0:
            hid_channels = aux_head["hid_channels"]
            self.aux_head = nn.Sequential(
                nn.Linear(in_features=in_channels, out_features=hid_channels),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=hid_channels, out_features=out_channels),
            )
        else:
            self.aux_head = nn.Linear(
                in_features=in_channels, out_features=out_channels
            )

    def forward_train(self, x, gt_labels):
        """
        Forward train head using the Supervised Contrastive Loss
        Args:
            x (Tensor): features from the backbone.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        losses = dict()
        fc_feats = self.fc(x)

        bsz = gt_labels.shape[0]
        # reshape aux_feats from [2 * bsz, dims] to [bs, 2, dims]
        f1, f2 = torch.split(self.aux_head(x), [bsz, bsz], dim=0)
        aux_feats = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss = self.compute_loss(aux_feats, gt_labels, fc_feats=fc_feats)
        losses.update(loss)
        return losses

    def simple_test(self, img):
        """
        Test without data augmentation.
        """
        cls_score = self.fc(img)

        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        pred = F.softmax(cls_score, dim=1) if cls_score is not None else None
        if torch.onnx.is_in_onnx_export():
            return pred
        pred = list(pred.detach().cpu().numpy())
        return pred
