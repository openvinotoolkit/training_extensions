# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX detection data transforms."""

from __future__ import annotations

from typing import Any

import torch
from torchvision import tv_tensors
from torchvision.ops.boxes import box_iou
from torchvision.transforms.v2 import RandomIoUCrop
from torchvision.transforms.v2 import functional as F  # noqa: N812
from torchvision.transforms.v2._utils import get_bounding_boxes, query_size


class MinIoURandomCrop(RandomIoUCrop):
    """MinIoURandomCrop inherited from RandomIoUCrop to align with mmdet.transforms.MinIoURandomCrop.

    * Updated
        - change `ious.max()` to `ious.min()` at L121 because both MinIoURandomCrop and v2.RandomIoUCrop seems similar,
          but MinIoURandomCrop uses `overlaps.min()` to check if there is at least one box smaller than `min_iou` (https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/datasets/transforms/transforms.py#L1432)
          and v2.RandomIoUCrop uses `ious.max()` to check if all boxes' IoUs are smaller than `min_jaccard_overlap` (https://github.com/pytorch/vision/blob/v0.16.1/torchvision/transforms/v2/_geometry.py#L1217).

        - `trials` in argument from 40 to 50 at L57.

    * Applied in other locations
        - box translation :
            mmdet : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/datasets/transforms/transforms.py#L1454
            torchvision : https://github.com/pytorch/vision/blob/v0.16.1/torchvision/transforms/v2/functional/_geometry.py#L1386

        - clip border :
            mmdet : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/datasets/transforms/transforms.py#L1456-L1457
            torchvision : https://github.com/pytorch/vision/blob/v0.16.1/torchvision/transforms/v2/functional/_geometry.py#L1389

        - except invalid bounding boxes :
            mmdet : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/datasets/transforms/transforms.py#L1453
            torchvision : https://github.com/pytorch/vision/blob/v0.16.1/torchvision/transforms/v2/_geometry.py#L1232-L1234
                + SanitizeBoundingBoxes

    Args:
        min_scale (float, optional): Minimum factors to scale the input size.
        max_scale (float, optional): Maximum factors to scale the input size.
        min_aspect_ratio (float, optional): Minimum aspect ratio for the cropped image or video.
        max_aspect_ratio (float, optional): Maximum aspect ratio for the cropped image or video.
        sampler_options (list of float, optional): List of minimal IoU (Jaccard) overlap between all the boxes and
            a cropped image or video. Default, ``None`` which corresponds to ``[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]``
        trials (int, optional): Number of trials to find a crop for a given value of minimal IoU (Jaccard) overlap.
            Default, 50.
    """

    def __init__(
        self,
        min_scale: float = 0.3,
        max_scale: float = 1.0,
        min_aspect_ratio: float = 0.5,
        max_aspect_ratio: float = 2.0,
        sampler_options: list[float] | None = None,
        trials: int = 50,  # 40 -> 50
    ):
        super().__init__(
            min_scale=min_scale,
            max_scale=max_scale,
            min_aspect_ratio=min_aspect_ratio,
            max_aspect_ratio=max_aspect_ratio,
            sampler_options=sampler_options,
            trials=trials,
        )

    def _get_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        orig_h, orig_w = query_size(flat_inputs)
        bboxes = get_bounding_boxes(flat_inputs)

        while True:
            # https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/datasets/transforms/transforms.py#L1407-L1410
            # sample an option
            idx = int(torch.randint(low=0, high=len(self.options), size=(1,)))
            min_jaccard_overlap = self.options[idx]
            if min_jaccard_overlap >= 1.0:  # a value larger than 1 encodes the leave as-is option
                return {}

            for _ in range(self.trials):
                # https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/datasets/transforms/transforms.py#L1414-L1419
                # check the aspect ratio limitations
                r = self.min_scale + (self.max_scale - self.min_scale) * torch.rand(2)
                new_w = int(orig_w * r[0])
                new_h = int(orig_h * r[1])
                aspect_ratio = new_w / new_h
                if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
                    continue

                # https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/datasets/transforms/transforms.py#L1421-L1428
                # check for 0 area crops
                r = torch.rand(2)
                left = int((orig_w - new_w) * r[0])
                top = int((orig_h - new_h) * r[1])
                right = left + new_w
                bottom = top + new_h
                if left == right or top == bottom:
                    continue

                # https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/datasets/transforms/transforms.py#L1439-L1445
                # https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/datasets/transforms/transforms.py#L1452-L1453
                # check for any valid boxes with centers within the crop area
                xyxy_bboxes = F.convert_bounding_box_format(
                    bboxes.as_subclass(torch.Tensor),
                    bboxes.format,
                    tv_tensors.BoundingBoxFormat.XYXY,
                )
                cx = 0.5 * (xyxy_bboxes[..., 0] + xyxy_bboxes[..., 2])
                cy = 0.5 * (xyxy_bboxes[..., 1] + xyxy_bboxes[..., 3])
                is_within_crop_area = (left < cx) & (cx < right) & (top < cy) & (cy < bottom)
                if not is_within_crop_area.any():
                    continue

                # https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/datasets/transforms/transforms.py#L1429-L1433
                xyxy_bboxes = xyxy_bboxes[is_within_crop_area]
                ious = box_iou(
                    xyxy_bboxes,
                    torch.tensor([[left, top, right, bottom]], dtype=xyxy_bboxes.dtype, device=xyxy_bboxes.device),
                )
                if ious.min() < min_jaccard_overlap:  # max -> min
                    continue

                return {
                    "top": top,
                    "left": left,
                    "height": new_h,
                    "width": new_w,
                    "is_within_crop_area": is_within_crop_area,
                }
