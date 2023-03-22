"""Module for defining ConvClsHead used for MMOV inference."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch.nn.functional as F
from mmcls.models.builder import HEADS
from mmcls.models.heads import ClsHead
from torch import nn


@HEADS.register_module()
class ConvClsHead(ClsHead):
    """Convolutional classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        init_cfg (dict | optional): The extra init config of layers.
            Defaults to use dict(type='Normal', layer='Linear', std=0.01).
    """

    def __init__(self, num_classes, in_channels, init_cfg=None, **kwargs):
        init_cfg = init_cfg if init_cfg else dict(type="Kaiming", layer=["Conv2d"])
        super().__init__(init_cfg=init_cfg, **kwargs)

        self.in_channels = in_channels
        self.num_classes = num_classes

        if self.num_classes <= 0:
            raise ValueError(f"num_classes={num_classes} must be a positive integer")

        self.conv = nn.Conv2d(self.in_channels, self.num_classes, (1, 1))

    def pre_logits(self, x):
        """Preprocess logits."""
        if isinstance(x, tuple):
            x = x[-1]
        return x

    def simple_test(self, cls_score, softmax=True, post_process=True):
        """Inference without augmentation.

        Args:
            x (tuple[Tensor]): The input features.
                Multi-stage inputs are acceptable but only the last stage will
                be used to classify. The shape of every item should be
                ``(num_samples, in_channels)``.
            softmax (bool): Whether to softmax the classification score.
            post_process (bool): Whether to do post processing the
                inference results. It will convert the output to a list.

        Returns:
            Tensor | list: The inference results.

                - If no post processing, the output is a tensor with shape
                  ``(num_samples, num_classes)``.
                - If post processing, the output is a multi-dimentional list of
                  float and the dimensions are ``(num_samples, num_classes)``.
        """
        x = self.pre_logits(cls_score)
        cls_score = self.conv(x).squeeze()

        if softmax:
            pred = F.softmax(cls_score, dim=1) if cls_score is not None else None
        else:
            pred = cls_score

        if post_process:
            return self.post_process(pred)
        return pred

    def forward(self, x):
        """Forward fuction of ConvClsHead class."""
        return self.simple_test(x)

    def forward_train(self, cls_score, gt_label, **kwargs):
        """Forward_train fuction of ConvClsHead class."""
        x = self.pre_logits(cls_score)
        cls_score = self.conv(x).squeeze()
        losses = self.loss(cls_score, gt_label, **kwargs)
        return losses
