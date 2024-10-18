# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Implementation modified from mmdet.structures.mask.mask_target.py.

Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/structures/mask/mask_target.py
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
from datumaro.components.annotation import Polygon
from otx.core.utils.mask_util import crop_and_resize_masks, crop_and_resize_polygons
from torch import Tensor
from torch.nn.modules.utils import _pair
from torchvision import tv_tensors


def mask_target(
    pos_proposals_list: list[Tensor],
    pos_assigned_gt_inds_list: list[Tensor],
    gt_masks_list: list[list[Polygon]] | list[tv_tensors.Mask],
    mask_size: int,
    meta_infos: list[dict],
) -> Tensor:
    """Compute mask target for positive proposals in multiple images.

    Args:
        pos_proposals_list (list[Tensor]): Positive proposals in multiple
            images, each has shape (num_pos, 4).
        pos_assigned_gt_inds_list (list[Tensor]): Assigned GT indices for each
            positive proposals, each has shape (num_pos,).
        gt_masks_list (list[list[Polygon]] or list[tv_tensors.Mask]): Ground truth masks of
            each image.
        mask_size (int): The mask size.
        meta_infos (list[dict]): Meta information of each image.

    Returns:
        Tensor: Mask target of each image, has shape (num_pos, w, h).
    """
    mask_size_list = [mask_size for _ in range(len(pos_proposals_list))]
    mask_targets = map(
        mask_target_single,  # type: ignore[arg-type]
        pos_proposals_list,
        pos_assigned_gt_inds_list,
        gt_masks_list,
        mask_size_list,
        meta_infos,
    )
    _mask_targets = list(mask_targets)
    if len(_mask_targets) > 0:
        _mask_targets = torch.cat(_mask_targets)
    return _mask_targets


def mask_target_single(
    pos_proposals: Tensor,
    pos_assigned_gt_inds: Tensor,
    gt_masks: list[Polygon] | tv_tensors.Mask,
    mask_size: list[int],
    meta_info: dict,
) -> Tensor:
    """Compute mask target for each positive proposal in the image."""
    mask_size = _pair(mask_size)
    if len(gt_masks) == 0:
        warnings.warn("No ground truth masks are provided!", stacklevel=2)
        return pos_proposals.new_zeros((0, *mask_size))

    if isinstance(gt_masks[0], Polygon):
        crop_and_resize = crop_and_resize_polygons
    elif isinstance(gt_masks, tv_tensors.Mask):
        crop_and_resize = crop_and_resize_masks
    else:
        warnings.warn("Unsupported ground truth mask type!", stacklevel=2)
        return pos_proposals.new_zeros((0, *mask_size))

    device = pos_proposals.device
    num_pos = pos_proposals.size(0)
    if num_pos > 0:
        proposals_np = pos_proposals.cpu().numpy()
        maxh, maxw = meta_info["img_shape"]
        proposals_np[:, [0, 2]] = np.clip(proposals_np[:, [0, 2]], 0, maxw)
        proposals_np[:, [1, 3]] = np.clip(proposals_np[:, [1, 3]], 0, maxh)
        pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()

        mask_targets = crop_and_resize(
            gt_masks,
            proposals_np,
            mask_size,  # type: ignore[arg-type]
            inds=pos_assigned_gt_inds,
            device=device,
        )
    else:
        mask_targets = pos_proposals.new_zeros((0, *mask_size))

    return mask_targets


def masks_to_boxes(masks: Tensor, dtype: torch.dtype) -> Tensor:
    """Compute the bounding boxes around the provided masks.

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=dtype, device=masks.device)
    x = torch.arange(0, w, dtype=dtype, device=masks.device)
    y, x = torch.meshgrid(y, x)

    x_mask = masks * x.unsqueeze(0)
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e4).flatten(1).min(-1)[0]

    y_mask = masks * y.unsqueeze(0)
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e4).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)
