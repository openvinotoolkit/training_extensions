"""Custom RPN head for OTX template."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from mmdet.models.builder import HEADS
from mmdet.models.dense_heads import RPNHead


@HEADS.register_module()
class CustomRPNHead(RPNHead):
    """RPN head.

    Args:
        in_channels (int): Number of channels in the input feature map.
        init_cfg (dict or list[dict], optional): Initialization config dict.
        num_convs (int): Number of convolution layers in the head. Default 1.
    """

    def forward_single(self, x):
        """Forward feature map of a single scale level."""
        rpn_cls_score, rpn_bbox_pred = super().forward_single(x)
        if rpn_cls_score.device.type == "hpu":
            rpn_cls_score = rpn_cls_score.cpu()
            rpn_bbox_pred = rpn_bbox_pred.cpu()
        return rpn_cls_score, rpn_bbox_pred
