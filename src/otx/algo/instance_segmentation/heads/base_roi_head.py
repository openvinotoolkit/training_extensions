# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Implementation modified from mmdet.models.roi_heads.base_roi_head.py.

Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/models/roi_heads/base_roi_head.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from otx.core.data.entity.instance_segmentation import InstanceSegBatchDataEntity

if TYPE_CHECKING:
    from torch import Tensor, nn

    from otx.algo.utils.mmengine_utils import InstanceData


class BaseRoIHead(nn.Module):
    """Base class for RoIHeads."""

    def __init__(
        self,
        bbox_roi_extractor: nn.Module,
        bbox_head: nn.Module,
        mask_roi_extractor: nn.Module,
        mask_head: nn.Module,
        assigner: nn.Module,
        sampler: nn.Module,
        mask_thr_binary: float = 0.5,
        max_per_img: int = 100,
        nms_iou_threshold: float = 0.5,
        score_thr: float = 0.05,
    ) -> None:
        super().__init__()

        self.bbox_roi_extractor = bbox_roi_extractor
        self.bbox_head = bbox_head
        self.mask_thr_binary = mask_thr_binary
        self.max_per_img = max_per_img
        self.nms_iou_threshold = nms_iou_threshold
        self.score_thr = score_thr

        self.mask_roi_extractor = mask_roi_extractor
        self.mask_head = mask_head
        self.bbox_assigner = assigner
        self.bbox_sampler = sampler

    @property
    def with_bbox(self) -> bool:
        """bool: whether the RoI head contains a `bbox_head`."""
        return hasattr(self, "bbox_head") and self.bbox_head is not None

    @property
    def with_mask(self) -> bool:
        """bool: whether the RoI head contains a `mask_head`."""
        return hasattr(self, "mask_head") and self.mask_head is not None

    @property
    def with_shared_head(self) -> bool:
        """bool: whether the RoI head contains a `shared_head`."""
        return hasattr(self, "shared_head") and self.shared_head is not None

    def predict(
        self,
        x: tuple[Tensor],
        rpn_results_list: list[InstanceData],
        entity: InstanceSegBatchDataEntity,
        rescale: bool = False,
    ) -> list[InstanceData]:
        """Forward the roi head and predict detection results on the features of the upstream network."""
        if not self.with_bbox:
            msg = "Bbox head must be implemented."
            raise NotImplementedError(msg)
        batch_img_metas = [
            {
                "img_id": img_info.img_idx,
                "img_shape": img_info.img_shape,
                "ori_shape": img_info.ori_shape,
                "scale_factor": img_info.scale_factor,
                "ignored_labels": img_info.ignored_labels,
            }
            for img_info in entity.imgs_info
        ]

        # If it has the mask branch, the bbox branch does not need
        # to be scaled to the original image scale, because the mask
        # branch will scale both bbox and mask at the same time.
        bbox_rescale = rescale if not self.with_mask else False
        results_list = self.predict_bbox(
            x,
            batch_img_metas,
            rpn_results_list,
            rescale=bbox_rescale,
        )

        if self.with_mask:
            results_list = self.predict_mask(x, batch_img_metas, results_list, rescale=rescale)

        return results_list
