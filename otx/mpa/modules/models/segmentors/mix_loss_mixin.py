# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
import torch.nn as nn


class MixLossMixin(object):
    def forward_train(self, img, img_metas, gt_semantic_seg, aux_img=None, **kwargs):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            aux_img (Tensor): Auxiliary images.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        if aux_img is not None:
            mix_loss_enabled = False
            mix_loss_cfg = self.train_cfg.get("mix_loss", None)
            if mix_loss_cfg is not None:
                mix_loss_enabled = mix_loss_cfg.get("enable", False)
            if mix_loss_enabled:
                self.train_cfg.mix_loss.enable = mix_loss_enabled

        if self.train_cfg.mix_loss.enable:
            img = torch.cat([img, aux_img], dim=0)
            gt_semantic_seg = torch.cat([gt_semantic_seg, gt_semantic_seg], dim=0)

        return super().forward_train(img, img_metas, gt_semantic_seg, **kwargs)
