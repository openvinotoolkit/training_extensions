# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""utils for detection task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mmcv.utils import ext_loader
from torchvision.ops import nms as tv_nms
from torchvision.ops import roi_align as tv_roi_align

if TYPE_CHECKING:
    from mmcv.ops.nms import NMSop
    from mmcv.ops.roi_align import RoIAlign

ext_module = ext_loader.load_ext("_ext", ["nms", "softnms", "nms_match", "nms_rotated", "nms_quadri"])


def monkey_patched_nms(
    ctx: NMSop,
    bboxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float,
    offset: float,
    score_threshold: float,
    max_num: int,
) -> torch.Tensor:
    """Runs MMCVs NMS with torchvision.nms, or forces NMS from MMCV to run on CPU."""
    _ = ctx
    is_filtering_by_score = score_threshold > 0
    if is_filtering_by_score:
        valid_mask = scores > score_threshold
        bboxes, scores = bboxes[valid_mask], scores[valid_mask]
        valid_inds = torch.nonzero(valid_mask, as_tuple=False).squeeze(dim=1)

    if bboxes.dtype == torch.bfloat16:
        bboxes = bboxes.to(torch.float32)
    if scores.dtype == torch.bfloat16:
        scores = scores.to(torch.float32)

    if offset == 0:
        inds = tv_nms(bboxes, scores, float(iou_threshold))
    else:
        device = bboxes.device
        bboxes = bboxes.to("cpu")
        scores = scores.to("cpu")
        inds = ext_module.nms(bboxes, scores, iou_threshold=float(iou_threshold), offset=offset)
        bboxes = bboxes.to(device)
        scores = scores.to(device)

    if max_num > 0:
        inds = inds[:max_num]
    if is_filtering_by_score:
        inds = valid_inds[inds]
    return inds


def monkey_patched_roi_align(self: RoIAlign, _input: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
    """Replaces MMCVs roi align with the one from torchvision.

    Args:
        self: patched instance
        _input: NCHW images
        rois: Bx5 boxes. First column is the index into N. The other 4 columns are xyxy.
    """
    if "aligned" in tv_roi_align.__code__.co_varnames:
        return tv_roi_align(_input, rois, self.output_size, self.spatial_scale, self.sampling_ratio, self.aligned)
    if self.aligned:
        rois -= rois.new_tensor([0.0] + [0.5 / self.spatial_scale] * 4)
    return tv_roi_align(_input, rois, self.output_size, self.spatial_scale, self.sampling_ratio)
