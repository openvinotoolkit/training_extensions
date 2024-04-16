"""The original source code is from mmdet. Please refer to https://github.com/open-mmlab/mmdetection/."""

# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import torch
from mmengine.registry import TASK_UTILS

from otx.algo.instance_segmentation.mmdet.structures.bbox import bbox_overlaps


def cast_tensor_type(x: torch.Tensor, scale: float = 1.0, dtype: str | None = None) -> torch.Tensor:
    """Cast tensor type to fp16."""
    if dtype == "fp16":
        # scale is for preventing overflows
        x = (x / scale).half()
    return x


@TASK_UTILS.register_module()
class BboxOverlaps2D:
    """2D Overlaps (e.g. IoUs, GIoUs) Calculator."""

    def __init__(self, scale: float = 1.0, dtype: str | None = None) -> None:
        self.scale = scale
        self.dtype = dtype

    def __call__(
        self,
        bboxes1: torch.Tensor,
        bboxes2: torch.Tensor,
        mode: str = "iou",
        is_aligned: bool = False,
    ) -> torch.Tensor:
        """Calculate IoU between 2D bboxes.

        Args:
            bboxes1 (Tensor or :obj:`BaseBoxes`): bboxes have shape (m, 4)
                in <x1, y1, x2, y2> format, or shape (m, 5) in <x1, y1, x2,
                y2, score> format.
            bboxes2 (Tensor or :obj:`BaseBoxes`): bboxes have shape (m, 4)
                in <x1, y1, x2, y2> format, shape (m, 5) in <x1, y1, x2, y2,
                score> format, or be empty. If ``is_aligned `` is ``True``,
                then m and n must be equal.
            mode (str): "iou" (intersection over union), "iof" (intersection
                over foreground), or "giou" (generalized intersection over
                union).
            is_aligned (bool, optional): If True, then m and n must be equal.
                Default False.

        Returns:
            Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
        """
        if bboxes1.size(-1) not in [0, 4, 5]:
            msg = "The last dimension of bboxes must be 4 or 5."
            raise ValueError(msg)
        if bboxes2.size(-1) not in [0, 4, 5]:
            msg = "The last dimension of bboxes must be 4 or 5."
            raise ValueError(msg)
        if bboxes2.size(-1) == 5:
            bboxes2 = bboxes2[..., :4]
        if bboxes1.size(-1) == 5:
            bboxes1 = bboxes1[..., :4]

        if self.dtype == "fp16":
            # change tensor type to save cpu and cuda memory and keep speed
            bboxes1 = cast_tensor_type(bboxes1, self.scale, self.dtype)
            bboxes2 = cast_tensor_type(bboxes2, self.scale, self.dtype)
            overlaps = bbox_overlaps(bboxes1, bboxes2, mode, is_aligned)
            if not overlaps.is_cuda and overlaps.dtype == torch.float16:
                # resume cpu float32
                overlaps = overlaps.float()
            return overlaps

        return bbox_overlaps(bboxes1, bboxes2, mode, is_aligned)
