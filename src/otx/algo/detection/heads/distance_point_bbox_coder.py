""""Distance Point BBox coder."""
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.

from __future__ import annotations

from typing import TYPE_CHECKING

from otx.algo.detection.utils.utils import bbox2distance, distance2bbox

if TYPE_CHECKING:
    from torch import Tensor


class DistancePointBBoxCoder:
    """Distance Point BBox coder.

    This coder encodes gt bboxes (x1, y1, x2, y2) into (top, bottom, left,
    right) and decode it back to the original.

    Args:
        clip_border (bool, optional): Whether clip the objects outside the
            border of the image. Defaults to True.
    """

    def __init__(
        self,
        clip_border: bool = True,
        encode_size: int = 4,
        use_box_type: bool = False,
    ) -> None:
        self.clip_border = clip_border
        self.encode_size = encode_size
        self.use_box_type = use_box_type

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
