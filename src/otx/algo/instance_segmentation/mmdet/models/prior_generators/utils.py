"""The original source code is from mmdet. Please refer to https://github.com/open-mmlab/mmdetection/."""

# TODO(Eugene): Revisit mypy errors after deprecation of mmlab
# https://github.com/openvinotoolkit/training_extensions/pull/3281
# mypy: ignore-errors
# ruff: noqa

# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

from otx.algo.instance_segmentation.mmdet.structures.bbox import BaseBoxes
from torch import Tensor


def anchor_inside_flags(
    flat_anchors: Tensor,
    valid_flags: Tensor,
    img_shape: Tuple[int],
    allowed_border: int = 0,
) -> Tensor:
    """Check whether the anchors are inside the border.

    Args:
        flat_anchors (torch.Tensor): Flatten anchors, shape (n, 4).
        valid_flags (torch.Tensor): An existing valid flags of anchors.
        img_shape (tuple(int)): Shape of current image.
        allowed_border (int): The border to allow the valid anchor.
            Defaults to 0.

    Returns:
        torch.Tensor: Flags indicating whether the anchors are inside a \
            valid range.
    """
    img_h, img_w = img_shape[:2]
    if allowed_border >= 0:
        if isinstance(flat_anchors, BaseBoxes):
            inside_flags = valid_flags & flat_anchors.is_inside(
                [img_h, img_w],
                all_inside=True,
                allowed_border=allowed_border,
            )
        else:
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
