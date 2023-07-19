"""Custom SOLOv2 Head of OTX Detection."""

# Copyright (C) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import torch
import torch.nn.functional as F
from mmdet.core import mask_matrix_nms
from mmdet.models.builder import HEADS
from mmdet.models.dense_heads import SOLOV2Head


@HEADS.register_module()
class CustomSOLOV2Head(SOLOV2Head):
    """Custom SOLOv2 Head of OTX Detection."""

    def _get_results_single(self, kernel_preds, cls_scores, mask_feats, img_meta, cfg=None):
        """Get processed mask related results of single image.

        Args:
            kernel_preds (Tensor): Dynamic kernel prediction of all points
                in single image, has shape
                (num_points, kernel_out_channels).
            cls_scores (Tensor): Classification score of all points
                in single image, has shape (num_points, num_classes).
            mask_feats (Tensor): Mask features of all points in
                single image, has shape (num_points, feat_h, feat_w).
            img_meta (dict): Meta information of corresponding image.
            cfg (dict, optional): Config used in test phase.
                Default: None.

        Returns:
            - scores (Tensor): Classification scores, has shape
                (num_instance,).
            - labels (Tensor): Has shape (num_instances,).
            - masks (Tensor): Processed mask results, has
                shape (num_instances, h, w).
        """
        if torch.onnx.is_in_onnx_export():
            return self._get_onnx_results_single(kernel_preds, cls_scores, mask_feats, img_meta, cfg)
        return super()._get_results_single(kernel_preds, cls_scores, mask_feats, img_meta, cfg)

    def _get_onnx_results_single(self, kernel_preds, cls_scores, mask_feats, img_meta, cfg=None):

        cfg = self.test_cfg if cfg is None else cfg
        assert len(kernel_preds) == len(cls_scores)

        featmap_size = mask_feats.size()[-2:]

        img_shape = img_meta["img_shape"]
        ori_shape = img_meta["ori_shape"]

        # overall info
        h, w = img_shape
        upsampled_size = (featmap_size[0] * self.mask_stride, featmap_size[1] * self.mask_stride)

        # process.
        score_mask = cls_scores > 0.0
        cls_scores = cls_scores[score_mask]

        # cate_labels & kernel_preds
        inds = score_mask.nonzero()
        cls_labels = inds[:, 1]
        kernel_preds = kernel_preds[inds[:, 0]]

        # trans vector.
        lvl_interval = cls_labels.new_tensor(self.num_grids).pow(2).cumsum(0)
        strides = kernel_preds.new_ones(lvl_interval[-1])

        strides[: lvl_interval[0]] *= self.strides[0]
        for lvl in range(1, self.num_levels):
            strides[lvl_interval[lvl - 1] : lvl_interval[lvl]] *= self.strides[lvl]
        strides = strides[inds[:, 0]]

        # mask encoding.
        # NOTE: ONNX does not support the following dynamic convolution:
        # kernel_preds = kernel_preds.view(
        #     kernel_preds.size(0), -1, self.dynamic_conv_size,
        #     self.dynamic_conv_size)
        # mask_preds = F.conv2d(
        #     mask_feats, kernel_preds, stride=1).squeeze(0).sigmoid()

        # NOTE: rewrite it with:
        assert self.dynamic_conv_size == 1, "ONNX does not support dynamic convolution"
        mask_feats = mask_feats.view(mask_feats.size(0), mask_feats.size(1), mask_feats.size(2) * mask_feats.size(3))
        mask_feats = mask_feats.permute(0, 2, 1)
        kernel_preds = kernel_preds.permute(1, 0)
        mask_preds = torch.matmul(mask_feats, kernel_preds)
        mask_preds = mask_preds.permute(0, 2, 1)
        mask_preds = mask_preds.view(mask_preds.size(0), mask_preds.size(1), featmap_size[0], featmap_size[1])
        mask_preds = mask_preds.squeeze(0).sigmoid()

        # mask.
        masks = mask_preds > cfg.mask_thr
        sum_masks = masks.sum((1, 2)).float()
        keep = sum_masks > strides
        masks = masks[keep]
        mask_preds = mask_preds[keep]
        sum_masks = sum_masks[keep]
        cls_scores = cls_scores[keep]
        cls_labels = cls_labels[keep]

        # maskness.
        mask_scores = (mask_preds * masks).sum((1, 2)) / sum_masks
        cls_scores *= mask_scores

        scores, labels, _, keep_inds = mask_matrix_nms(
            masks,
            cls_labels,
            cls_scores,
            mask_area=sum_masks,
            nms_pre=cfg.nms_pre,
            max_num=cfg.max_per_img,
            kernel=cfg.kernel,
            sigma=cfg.sigma,
            filter_thr=0.0,
        )
        mask_preds = mask_preds[keep_inds]
        mask_preds = F.interpolate(mask_preds.unsqueeze(0), size=upsampled_size, mode="bilinear", align_corners=False)[
            :, :, :h, :w
        ]
        mask_preds = F.interpolate(mask_preds, size=ori_shape[:2], mode="bilinear", align_corners=False).squeeze(0)
        masks = mask_preds > cfg.mask_thr
        return masks, labels, scores
