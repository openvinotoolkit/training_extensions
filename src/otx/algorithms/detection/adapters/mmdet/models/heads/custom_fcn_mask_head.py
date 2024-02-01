"""CustomFCNMaskHead for OTX template."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
from mmdet.models.builder import HEADS
from mmdet.models.roi_heads.mask_heads.fcn_mask_head import FCNMaskHead

from otx.algorithms.common.adapters.mmdeploy.utils import is_mmdeploy_enabled


@HEADS.register_module()
class CustomFCNMaskHead(FCNMaskHead):
    """Custom FCN Mask Head for fast mask evaluation."""

    def get_seg_masks(self, mask_pred, det_bboxes, det_labels, rcnn_test_cfg, ori_shape, scale_factor, rescale):
        """Get segmentation masks from mask_pred and bboxes.

        The original `FCNMaskHead.get_seg_masks` grid sampled 28 x 28 masks to the original image resolution.
        As a result, the resized masks occupy a large amount of memory and slow down the inference.
        This method directly returns 28 x 28 masks and resize to bounding boxes size in post-processing step.
        Doing so can save memory and speed up the inference.

        Args:
            mask_pred (Tensor or ndarray): shape (n, #class, h, w).
                For single-scale testing, mask_pred is the direct output of
                model, whose type is Tensor, while for multi-scale testing,
                it will be converted to numpy array outside of this method.
            det_bboxes (Tensor): shape (n, 4/5)
            det_labels (Tensor): shape (n, )
            rcnn_test_cfg (dict): rcnn testing config
            ori_shape (Tuple): original image height and width, shape (2,)
            scale_factor(ndarray | Tensor): If ``rescale is True``, box
                coordinates are divided by this scale factor to fit
                ``ori_shape``.
            rescale (bool): If True, the resulting masks will be rescaled to
                ``ori_shape``.

        Returns:
            list[list]: encoded masks. The c-th item in the outer list
                corresponds to the c-th class. Given the c-th outer list, the
                i-th item in that inner list is the mask for the i-th box with
                class label c.

        """
        if isinstance(mask_pred, torch.Tensor):
            mask_pred = mask_pred.sigmoid()
        else:
            # In AugTest, has been activated before
            mask_pred = det_bboxes.new_tensor(mask_pred)

        cls_segms = [[] for _ in range(self.num_classes)]  # BG is not included in num_classes
        labels = det_labels

        N = len(mask_pred)
        # The actual implementation split the input into chunks,
        # and paste them chunk by chunk.

        threshold = rcnn_test_cfg.mask_thr_binary

        if not self.class_agnostic:
            mask_pred = mask_pred[range(N), labels][:, None]

        for i in range(N):
            mask = mask_pred[i]
            if threshold >= 0:
                mask = (mask >= threshold).to(dtype=torch.bool)
            else:
                # for visualization and debugging
                mask = (mask * 255).to(dtype=torch.uint8)
            mask = mask.detach().cpu().numpy()
            cls_segms[labels[i]].append(mask[0])
        return cls_segms

    def get_scaled_seg_masks(self, *args, **kwargs):
        """Original method "get_seg_mask" from FCNMaskHead. Used in Semi-SL algorithm."""
        return super().get_seg_masks(*args, **kwargs)


if is_mmdeploy_enabled():
    from mmdeploy.core import FUNCTION_REWRITER

    @FUNCTION_REWRITER.register_rewriter(
        "otx.algorithms.detection.adapters.mmdet.models." "heads.custom_fcn_mask_head.CustomFCNMaskHead.get_seg_masks"
    )
    def custom_fcn_mask_head__get_seg_masks(
        ctx, self, mask_pred, det_bboxes, det_labels, rcnn_test_cfg, ori_shape, **kwargs
    ):
        """Rewrite `get_seg_masks` of `FCNMaskHead` for default backend.

        Rewrite the get_seg_masks for only fcn_mask_head inference.

        Args:
            ctx (dict): context dict
            self (CustomFCNMaskHead): CustomFCNMaskHead instance
            mask_pred (Tensor): shape (n, #class, h, w).
            det_bboxes (Tensor): shape (n, 4/5)
            det_labels (Tensor): shape (n, )
            rcnn_test_cfg (dict): rcnn testing config
            ori_shape (Tuple): original image height and width, shape (2,)
            kwargs (dict): other arguments

        Returns:
            Tensor: a mask of shape (N, img_h, img_w).
        """
        mask_pred = mask_pred.sigmoid()
        bboxes = det_bboxes[:, :4]
        labels = det_labels
        if not self.class_agnostic:
            box_inds = torch.arange(mask_pred.shape[0], device=bboxes.device)
            mask_pred = mask_pred[box_inds, labels][:, None]
        return mask_pred
