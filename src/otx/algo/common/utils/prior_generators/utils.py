# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Utils for anchor generators.

Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/models/task_modules/prior_generators/utils.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


def anchor_inside_flags(
    flat_anchors: Tensor,
    valid_flags: Tensor,
    img_shape: tuple[int, ...],
    allowed_border: int = 0,
) -> Tensor:
    """Check whether the anchors are inside the border.

    Args:
        flat_anchors (Tensor): Flatten anchors, shape (n, 4).
        valid_flags (Tensor): An existing valid flags of anchors.
        img_shape (tuple(int)): Shape of current image.
        allowed_border (int): The border to allow the valid anchor.
            Defaults to 0.

    Returns:
        Tensor: Flags indicating whether the anchors are inside a \
            valid range.
    """
    img_h, img_w = img_shape[:2]
    if allowed_border >= 0:
        inside_flags = (
            valid_flags
            & (flat_anchors[:, 0] >= -allowed_border)
            & (flat_anchors[:, 1] >= -allowed_border)
            & (flat_anchors[:, 2] < img_w + allowed_border)
            & (flat_anchors[:, 3] < img_h + allowed_border)
        )
    else:
        inside_flags = valid_flags
    return inside_flags
