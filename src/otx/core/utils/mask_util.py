# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Utility functions for mask operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pycocotools.mask as mask_utils
import torch
from datumaro import Polygon
from torchvision.ops import roi_align

if TYPE_CHECKING:
    from torchvision import tv_tensors


def polygon_to_bitmap(
    polygons: list[Polygon],
    height: int,
    width: int,
) -> np.ndarray:
    """Convert a list of polygons to a bitmap mask.

    Args:
        polygons (list[Polygon]): List of Datumaro Polygon objects.
        height (int): bitmap height
        width (int): bitmap width

    Returns:
        np.ndarray: bitmap masks
    """
    polygons = [polygon.points for polygon in polygons]
    rles = mask_utils.frPyObjects(polygons, height, width)
    return mask_utils.decode(rles).astype(bool).transpose((2, 0, 1))


def polygon_to_rle(
    polygons: list[Polygon],
    height: int,
    width: int,
) -> list[dict]:
    """Convert a list of polygons to a list of RLE masks.

    Args:
        polygons (list[Polygon]): List of Datumaro Polygon objects.
        height (int): bitmap height
        width (int): bitmap width

    Returns:
        list[dict]: List of RLE masks.
    """
    polygons = [polygon.points for polygon in polygons]
    if len(polygons):
        return mask_utils.frPyObjects(polygons, height, width)
    return []


def encode_rle(mask: torch.Tensor) -> dict:
    """Encodes a mask into RLE format.

    Rewrite of https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py

    Example:
        Given M=[0 0 1 1 1 0 1] the RLE counts is [2 3 1 1].
        Or for M=[1 1 1 1 1 1 0] the RLE counts is [0 6 1].

    Args:
        mask (torch.Tensor): A binary mask (0 or 1) of shape (H, W).

    Returns:
        dict: A dictionary with keys "counts" and "size".
    """
    device = mask.device
    vector = mask.t().ravel()
    diffs = torch.diff(vector)
    next_diffs = torch.where(diffs != 0)[0] + 1

    counts = torch.diff(
        torch.cat(
            (
                torch.tensor([0], device=device),
                next_diffs,
                torch.tensor([len(vector)], device=device),
            ),
        ),
    )

    # odd counts are always the numbers of zeros
    if vector[0] == 1:
        counts = torch.cat((torch.tensor([0], device=device), counts))

    return {"counts": counts.tolist(), "size": list(mask.shape)}


def crop_and_resize_polygons(
    annos: list[Polygon],
    bboxes: np.ndarray,
    out_shape: tuple,
    inds: np.ndarray,
    device: str = "cpu",
) -> torch.Tensor:
    """Crop and resize polygons to the target size."""
    out_h, out_w = out_shape
    if len(annos) == 0:
        return torch.empty((0, *out_shape), dtype=torch.float, device=device)

    resized_polygons = []
    for i in range(len(bboxes)):
        polygon = annos[inds[i]]
        bbox = bboxes[i, :]
        x1, y1, x2, y2 = bbox
        w = np.maximum(x2 - x1, 1)
        h = np.maximum(y2 - y1, 1)
        h_scale = out_h / max(h, 0.1)  # avoid too large scale
        w_scale = out_w / max(w, 0.1)

        points = polygon.points
        points = points.copy()
        points = np.array(points)
        # crop
        # pycocotools will clip the boundary
        points[0::2] = points[0::2] - bbox[0]
        points[1::2] = points[1::2] - bbox[1]

        # resize
        points[0::2] = points[0::2] * w_scale
        points[1::2] = points[1::2] * h_scale

        resized_polygon = Polygon(points.tolist())

        resized_polygons.append(resized_polygon)

    mask_targets = polygon_to_bitmap(resized_polygons, *out_shape)

    return torch.from_numpy(mask_targets).float().to(device)


def crop_and_resize_masks(
    annos: tv_tensors.Mask,
    bboxes: np.ndarray,
    out_shape: tuple,
    inds: np.ndarray,
    device: str = "cpu",
) -> torch.Tensor:
    """Crop and resize masks to the target size."""
    if len(annos) == 0:
        return torch.empty((0, *out_shape), dtype=torch.float, device=device)

    # convert bboxes to tensor
    if isinstance(bboxes, np.ndarray):
        bboxes = torch.from_numpy(bboxes).to(device=device)
    if isinstance(inds, np.ndarray):
        inds = torch.from_numpy(inds).to(device=device)

    num_bbox = bboxes.shape[0]
    fake_inds = torch.arange(num_bbox, device=device).to(dtype=bboxes.dtype)[:, None]
    rois = torch.cat([fake_inds, bboxes], dim=1)  # Nx5
    rois = rois.to(device=device)
    if num_bbox > 0:
        gt_masks_th = annos.index_select(0, inds).to(dtype=rois.dtype)
        targets = roi_align(gt_masks_th[:, None, :, :], rois, out_shape, 1.0, 0, True).squeeze(1)
        resized_masks = targets >= 0.5
    else:
        resized_masks = torch.empty((0, *out_shape), device=device)
    return resized_masks.float()
