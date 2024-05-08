# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# This class and its supporting functions are adapted from the mmdet.
# Please refer to https://github.com/open-mmlab/mmdetection/
"""MMDet Mask Structure."""

from __future__ import annotations

import numpy as np
import torch
from datumaro.components.annotation import Polygon
from otx.core.utils.mask_util import crop_and_resize_polygons, polygon_to_bitmap
from torch.nn.modules.utils import _pair


def mask_target(
    pos_proposals_list: list[torch.Tensor],
    pos_assigned_gt_inds_list: list[torch.Tensor],
    gt_masks_list: list[list[Polygon]],
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
    gt_masks: list[Polygon],
    cfg: dict,
) -> torch.Tensor:
    """Compute mask target for each positive proposal in the image."""
    if not isinstance(gt_masks[0], Polygon):
        msg = "Mask target only supports polygon format masks"
        raise TypeError(msg)

    device = pos_proposals.device
    mask_size = _pair(cfg["mask_size"])
    num_pos = pos_proposals.size(0)
    if num_pos > 0:
        proposals_np = pos_proposals.cpu().numpy()
        maxh, maxw = gt_masks[0].attributes["img_shape"]
        proposals_np[:, [0, 2]] = np.clip(proposals_np[:, [0, 2]], 0, maxw)
        proposals_np[:, [1, 3]] = np.clip(proposals_np[:, [1, 3]], 0, maxh)
        pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()

        polygon_targets = crop_and_resize_polygons(
            gt_masks,
            proposals_np,
            mask_size,
            inds=pos_assigned_gt_inds,
        )

        polygon_targets = polygon_to_bitmap(polygon_targets, *mask_size)

        polygon_targets = torch.from_numpy(polygon_targets).float().to(device)
    else:
        polygon_targets = pos_proposals.new_zeros((0, *mask_size))

    return polygon_targets
