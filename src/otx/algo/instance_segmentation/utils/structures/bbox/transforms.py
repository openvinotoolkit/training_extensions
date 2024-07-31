# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Implementation modified from mmdet.structures.bbox.transforms.py.

Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/structures/bbox/transforms.py
"""

from __future__ import annotations

import torch
from torch import Tensor


def bbox2roi(bbox_list: list[Tensor]) -> Tensor:
    """Convert a list of bboxes to roi format.

    Args:
        bbox_list (list[Union[Tensor]): a list of bboxes corresponding to a batch of images.

    Returns:
        Tensor: shape (n, box_dim + 1), where ``box_dim`` depends on the
        different box types. For example, If the box type in ``bbox_list``
        is HorizontalBoxes, the output shape is (n, 5). Each row of data
        indicates [batch_ind, x1, y1, x2, y2].
    """
    rois_list = []
    for img_id, bboxes in enumerate(bbox_list):
        img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
        rois = torch.cat([img_inds, bboxes], dim=-1)
        rois_list.append(rois)
    return torch.cat(rois_list, 0)


def scale_boxes(boxes: Tensor, scale_factor: list[float]) -> Tensor:
    """Scale boxes with type of tensor or box type.

    Args:
        boxes (Tensor): boxes need to be scaled. Its type
            can be a tensor or a box type.
        scale_factor (tuple[float, float]): factors for scaling boxes.
            The length should be 2.

    Returns:
        Tensor: Scaled boxes.
    """
    # Tensor boxes will be treated as horizontal boxes
    repeat_num = int(boxes.size(-1) / 2)
    scale_factor = boxes.new_tensor(scale_factor[::-1]).repeat((1, repeat_num))
    return boxes * scale_factor


def get_box_wh(boxes: Tensor) -> tuple[Tensor, Tensor]:
    """Get the width and height of boxes with type of tensor or box type.

    Args:
        boxes (Tensor): boxes with type of tensor or box type.

    Returns:
        tuple[Tensor, Tensor]: the width and height of boxes.
    """
    # Tensor boxes will be treated as horizontal boxes by defaults
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    return w, h


def empty_box_as(boxes: Tensor) -> Tensor:
    """Generate empty box according to input ``boxes` type and device.

    Args:
        boxes (Tensor): boxes with type of tensor or box type.

    Returns:
        Tensor: Generated empty box.
    """
    return boxes.new_zeros(0, 4)
