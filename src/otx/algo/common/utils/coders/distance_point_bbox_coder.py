# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Implementation modified from mmdet.models.task_modules.coders.distance_point_bbox_coder.

Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/models/task_modules/coders/distance_point_bbox_coder.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from otx.algo.common.utils.utils import bbox2distance, distance2bbox
from otx.algo.detection.utils.utils import distance2bbox_export

if TYPE_CHECKING:
    from torch import Tensor


class DistancePointBBoxCoder:
    """Distance Point BBox coder.

    This coder encodes gt bboxes (x1, y1, x2, y2) into (top, bottom, left,
    right) and decode it back to the original.

    Args:
        clip_border (bool, optional): Whether clip the objects outside the
            border of the image. Defaults to True.
        encode_size (int): Encode size.
    """

    def __init__(self, clip_border: bool = True, encode_size: int = 4) -> None:
        self.clip_border = clip_border
        self.encode_size = encode_size

    def encode(
        self,
        points: Tensor,
        gt_bboxes: Tensor,
        max_dis: float | None = None,
        eps: float = 0.1,
    ) -> Tensor:
        """Encode bounding box to distances.

        Args:
            points (Tensor): Shape (N, 2), The format is [x, y].
            gt_bboxes (Tensor or :obj:`BaseBoxes`): Shape (N, 4), The format
                is "xyxy"
            max_dis (float): Upper bound of the distance. Default None.
            eps (float): a small value to ensure target < max_dis, instead <=.
                Default 0.1.

        Returns:
            Tensor: Box transformation deltas. The shape is (N, 4).
        """
        if points.size(0) != gt_bboxes.size(0):
            msg = "The number of points should be equal to the number of boxes."
            raise ValueError(msg)
        if points.size(-1) != 2:
            msg = "The last dimension of points should be 2."
            raise ValueError(msg)
        if gt_bboxes.size(-1) != 4:
            msg = "The last dimension of gt_bboxes should be 4."
            raise ValueError(msg)
        return bbox2distance(points, gt_bboxes, max_dis, eps)

    def decode(
        self,
        points: Tensor,
        pred_bboxes: Tensor,
        max_shape: tuple[int, ...] | Tensor | tuple[tuple[int, ...], ...] | None = None,
    ) -> Tensor:
        """Decode distance prediction to bounding box."""
        if points.size(0) != pred_bboxes.size(0):
            msg = "The number of points should be equal to the number of boxes."
            raise ValueError(msg)
        if points.size(-1) != 2:
            msg = "The last dimension of points should be 2."
            raise ValueError(msg)
        if pred_bboxes.size(-1) != 4:
            msg = "The last dimension of pred_bboxes should be 4."
            raise ValueError(msg)
        if self.clip_border is False:
            max_shape = None
        return distance2bbox(points, pred_bboxes, max_shape)

    def decode_export(
        self,
        points: Tensor,
        pred_bboxes: Tensor,
        max_shape: tuple[int, ...] | Tensor | tuple[tuple[int, ...], ...] | None = None,
    ) -> Tensor:
        """Decode distance prediction to bounding box for export.

        Reference : https://github.com/open-mmlab/mmdeploy/blob/v1.3.1/mmdeploy/codebase/mmdet/models/task_modules/coders/distance_point_bbox_coder.py#L11-L42
        """
        if points.size(0) != pred_bboxes.size(0):
            msg = (
                f"The batch of points (={points.size(0)}) and the batch of pred_bboxes "
                f"(={pred_bboxes.size(0)}) should be same."
            )
            raise ValueError(msg)

        if points.size(-1) != 2:
            msg = f"points should have the format with size of 2, given {points.size(-1)}."
            raise ValueError(msg)

        if pred_bboxes.size(-1) != 4:
            msg = f"pred_bboxes should have the format with size of 4, given {pred_bboxes.size(-1)}."
            raise ValueError(msg)

        if self.clip_border is False:
            max_shape = None

        return distance2bbox_export(points, pred_bboxes, max_shape)
