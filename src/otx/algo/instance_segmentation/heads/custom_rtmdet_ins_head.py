# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Custom RTMDetInsSepBNHead for OTX RTMDet-InstSeg instance segmentation models."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812
from mmcv.ops import batched_nms
from mmdet.models.dense_heads.rtmdet_ins_head import RTMDetInsSepBNHead
from mmdet.registry import MODELS
from mmdet.structures.bbox import get_box_tensor, get_box_wh, scale_boxes
from mmengine.config import ConfigDict

if TYPE_CHECKING:
    from mmengine.structures import InstanceData


@MODELS.register_module()
class CustomRTMDetInsSepBNHead(RTMDetInsSepBNHead):
    """Custom RTMDet instance segmentation head."""

    def mask_postprocess(
        self,
        mask_logits: torch.Tensor,
        img_h: int,
        img_w: int,
        gpu_mem_limit: float = 1.0,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """Postprocess mask logits to binary masks.

        Args:
            masks (_type_): Mask Logits with shape (B, N, H, W).
            img_h (int): Image height resizes to.
            img_w (int): Image width resizes to.
            gpu_mem_limit (float, optional): GPU memory limit in GB. Defaults to 1.0.
            threshold (float, optional): Threshold for binary masks. Defaults to 0.5.

        Returns:
            torch.Tensor: Binary masks with shape (B, N, img_h, img_w).
        """
        masks = torch.zeros(
            size=(mask_logits.shape[:2] + (img_h, img_w)),
            dtype=torch.bool,
            device=mask_logits.device,
        )

        total_bytes = mask_logits.element_size() * masks.nelement()
        num_chunks = int(math.ceil(total_bytes / (gpu_mem_limit * 1024) ** 3))
        n = mask_logits.shape[1]
        chunks = torch.chunk(
            torch.arange(n, device=mask_logits.device),
            num_chunks,
        )
        for inds in chunks:
            masks_chunk = F.interpolate(
                mask_logits[:, inds],
                size=[
                    img_w,
                    img_h,
                ],
                mode="bilinear",
                align_corners=False,
            )
            masks[:, inds] = (masks_chunk >= threshold).to(dtype=torch.bool)
        return masks

    def _bbox_mask_post_process(
        self,
        results: InstanceData,
        mask_feat: torch.Tensor,
        cfg: ConfigDict | dict,
        rescale: bool = False,
        with_nms: bool = True,
        img_meta: dict | None = None,
    ) -> InstanceData:
        """Bbox and mask post-processing method.

        The boxes would be rescaled to the original image scale and do
        the nms operation. Usually `with_nms` is False is used for aug test.

        Args:
            results (:obj:`InstaceData`): Detection instance results,
                each item has shape (num_bboxes, ).
            mask_feat (Tensor): Mask prototype features of a single image
            cfg (ConfigDict): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default to False.
            with_nms (bool): If True, do nms before return boxes.
                Default to True.
            img_meta (dict, optional): Image meta info. Defaults to None.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.
                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, h, w).
        """
        if img_meta is None:
            img_meta = {}
            img_meta["scale_factor"] = [1.0, 1.0]

        if cfg is None:
            cfg = ConfigDict(
                nms_pre=300,
                mask_thr_binary=0.5,
                max_per_img=100,
                score_thr=0.05,
                nms=ConfigDict(type="nms", iou_threshold=0.6),
                min_bbox_size=10,
            )

        stride = self.prior_generator.strides[0][0]
        if rescale:
            scale_factor = [1 / s for s in img_meta["scale_factor"]]
            results.bboxes = scale_boxes(results.bboxes, scale_factor)

        if hasattr(results, "score_factors"):
            score_factors = results.pop("score_factors")
            results.scores = results.scores * score_factors

        # filter small size bboxes
        if cfg.min_bbox_size >= 0:
            w, h = get_box_wh(results.bboxes)
            valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
            if not valid_mask.all():
                results = results[valid_mask]

        if not with_nms:
            msg = "with_nms must be True for RTMDet-Ins"
            raise RuntimeError(msg)

        if results.bboxes.numel() > 0:
            bboxes = get_box_tensor(results.bboxes)
            det_bboxes, keep_idxs = batched_nms(bboxes, results.scores, results.labels, cfg.nms)
            results = results[keep_idxs]
            # some nms would reweight the score, such as softnms
            results.scores = det_bboxes[:, -1]
            results = results[: cfg.max_per_img]

            # process masks
            mask_logits = self._mask_predict_by_feat_single(mask_feat, results.kernels, results.priors)
            mask_logits = F.interpolate(mask_logits.unsqueeze(0), scale_factor=stride, mode="bilinear")

            if rescale:
                ori_h, ori_w = img_meta["ori_shape"][:2]
                masks = self.mask_postprocess(
                    mask_logits,
                    math.ceil(mask_logits.shape[-1] * scale_factor[1]),
                    math.ceil(mask_logits.shape[-2] * scale_factor[0]),
                    threshold=cfg.mask_thr_binary,
                )[..., :ori_h, :ori_w]
                masks = masks.squeeze(0)
            else:
                masks = mask_logits.sigmoid().squeeze(0)
                masks = masks > cfg.mask_thr_binary
            results.masks = masks
        else:
            h, w = img_meta["ori_shape"][:2] if rescale else img_meta["img_shape"][:2]
            results.masks = torch.zeros(
                size=(results.bboxes.shape[0], h, w),
                dtype=torch.bool,
                device=results.bboxes.device,
            )

        return results
