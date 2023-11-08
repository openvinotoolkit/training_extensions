"""CustomFCNMaskHead for OTX template."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import torch
from mmdet.models.roi_heads.mask_heads.fcn_mask_head import FCNMaskHead
from mmdet.registry import MODELS
from mmengine.config import ConfigDict
from torch import Tensor


@MODELS.register_module()
class CustomFCNMaskHead(FCNMaskHead):
    """Custom FCN Mask Head for fast mask evaluation."""

    def _predict_by_feat_single(
        self,
        mask_preds: Tensor,
        bboxes: Tensor,
        labels: Tensor,
        img_meta: dict,
        rcnn_test_cfg: ConfigDict,
        rescale: bool = False,
        activate_map: bool = False,
    ) -> Tensor:
        """Get segmentation masks from mask_preds and bboxes.

        The original `FCNMaskHead._predict_by_feat_single` grid sampled 28 x 28 masks to the original image resolution.
        As a result, the resized masks occupy a large amount of memory and slow down the inference.
        This method directly returns 28 x 28 masks and resize to bounding boxes size in post-processing step.
        Doing so can save memory and speed up the inference.

        Args:
            mask_preds (Tensor): Predicted foreground masks, has shape
                (n, num_classes, h, w).
            bboxes (Tensor): Predicted bboxes, has shape (n, 4)
            labels (Tensor): Labels of bboxes, has shape (n, )
            img_meta (dict): image information.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of Bbox Head.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            activate_map (book): Whether get results with augmentations test.
                If True, the `mask_preds` will not process with sigmoid.
                Defaults to False.

        Returns:
            Tensor: Encoded masks, has shape (n, img_w, img_h)
        """
        scale_factor = bboxes.new_tensor(img_meta["scale_factor"]).repeat((1, 2))
        img_h, img_w = img_meta["ori_shape"][:2]

        mask_preds = mask_preds.sigmoid() if not activate_map else bboxes.new_tensor(mask_preds)

        if rescale:  # in-placed rescale the bboxes
            bboxes /= scale_factor
        else:
            w_scale, h_scale = scale_factor[0, 0], scale_factor[0, 1]
            img_h = np.round(img_h * h_scale.item()).astype(np.int32)
            img_w = np.round(img_w * w_scale.item()).astype(np.int32)

        number_of_preds = len(mask_preds)
        # The actual implementation split the input into chunks,
        # and paste them chunk by chunk.

        threshold = rcnn_test_cfg.mask_thr_binary

        if not self.class_agnostic:
            mask_preds = mask_preds[range(number_of_preds), labels][:, None]

        masks = []
        for i in range(number_of_preds):
            mask = mask_preds[i]
            mask = (mask >= threshold).to(dtype=torch.bool) if threshold >= 0 else (mask * 255).to(dtype=torch.uint8)
            mask = mask.detach().cpu().numpy()
            masks.append(mask)
        return masks

    def get_scaled_seg_masks(self, *args, **kwargs) -> Tensor:
        """Original method "get_seg_mask" from FCNMaskHead. Used in Semi-SL algorithm."""
        return super()._predict_by_feat_single(*args, **kwargs)
