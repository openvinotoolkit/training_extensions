# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# This class and its supporting functions are adapted from the mmdet.
# Please refer to https://github.com/open-mmlab/mmdetection/
"""MMDet Mask Structure."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.nn.modules.utils import _pair

if TYPE_CHECKING:
    from mmdet.structures.mask import BitmapMasks


def mask_target(
    pos_proposals_list: list[torch.Tensor],
    pos_assigned_gt_inds_list: list[torch.Tensor],
    gt_masks_list: list[BitmapMasks],
    cfg: dict,
) -> torch.Tensor:
    """Compute mask target for positive proposals in multiple images.

    Args:
        pos_proposals_list (list[Tensor]): Positive proposals in multiple
            images, each has shape (num_pos, 4).
        pos_assigned_gt_inds_list (list[Tensor]): Assigned GT indices for each
            positive proposals, each has shape (num_pos,).
        gt_masks_list (list[:obj:`BaseInstanceMasks`]): Ground truth masks of
            each image.
        cfg (dict): Config dict that specifies the mask size.

    Returns:
        Tensor: Mask target of each image, has shape (num_pos, w, h).
    """
    cfg_list = [cfg for _ in range(len(pos_proposals_list))]
    mask_targets = map(mask_target_single, pos_proposals_list, pos_assigned_gt_inds_list, gt_masks_list, cfg_list)
    _mask_targets = list(mask_targets)
    if len(_mask_targets) > 0:
        _mask_targets = torch.cat(_mask_targets)
    return _mask_targets


def mask_target_single(
    pos_proposals: torch.Tensor,
    pos_assigned_gt_inds: torch.Tensor,
    gt_masks: BitmapMasks,
    cfg: dict,
) -> torch.Tensor | np.ndarray:
    """Compute mask target for each positive proposal in the image.

    Args:
        pos_proposals (Tensor): Positive proposals.
        pos_assigned_gt_inds (Tensor): Assigned GT inds of positive proposals.
        gt_masks (:obj:`BaseInstanceMasks`): GT masks in the format of Bitmap
            or Polygon.
        cfg (dict): Config dict that indicate the mask size.

    Returns:
        Tensor: Mask target of each positive proposals in the image.
    """
    device = pos_proposals.device
    mask_size = _pair(cfg["mask_size"])
    binarize = not cfg.get("soft_mask_target", False)
    num_pos = pos_proposals.size(0)
    if num_pos > 0:
        proposals_np = pos_proposals.cpu().numpy()
        maxh, maxw = gt_masks.height, gt_masks.width
        proposals_np[:, [0, 2]] = np.clip(proposals_np[:, [0, 2]], 0, maxw)
        proposals_np[:, [1, 3]] = np.clip(proposals_np[:, [1, 3]], 0, maxh)
        pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()

        mask_targets = gt_masks.crop_and_resize(
            proposals_np,
            mask_size,
            device=device,
            inds=pos_assigned_gt_inds,
            binarize=binarize,
        ).to_ndarray()

        mask_targets = torch.from_numpy(mask_targets).float().to(device)
    else:
        mask_targets = pos_proposals.new_zeros((0, *mask_size))

    return mask_targets
