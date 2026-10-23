# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for mask operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pycocotools.mask as mask_utils
import torch
from torchvision.ops import roi_align

from getitune.data.entity import ImageInfo, rasterize

if TYPE_CHECKING:
    import numpy as np
    from torchvision import tv_tensors


def polygon_to_bitmap(
    polygons: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    """Convert polygons to a bitmap mask.

    Args:
        polygons: a ragged array containing np.ndarray objects of shape (Npoly, 2)
        height: bitmap height
        width: bitmap width

    Returns:
        np.ndarray: bitmap masks
    """
    # Convert to list of flat point arrays for pycocotools
    polygon_points = [points.reshape(-1) for points in polygons]
    rles = mask_utils.frPyObjects(polygon_points, height, width)
    return mask_utils.decode(rles).astype(bool).transpose((2, 0, 1))


def polygon_to_rle(
    polygons: np.ndarray,
    height: int,
    width: int,
) -> list[dict]:
    """Convert polygons to a list of RLE masks.

    Args:
        polygons: a ragged array containing np.ndarray objects of shape (Npoly, 2)
        height: bitmap height
        width: bitmap width

    Returns:
        list[dict]: List of RLE masks.
    """
    # Convert to list of flat point arrays for pycocotools
    polygon_points = [points.reshape(-1) for points in polygons]

    return mask_utils.frPyObjects(polygon_points, height, width)


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

    bboxes_tensor = torch.from_numpy(bboxes).to(device=device)
    inds_tensor = torch.from_numpy(inds).to(device=device)

    num_bbox = bboxes_tensor.shape[0]
    fake_inds = torch.arange(num_bbox, device=device).to(dtype=bboxes_tensor.dtype)[:, None]
    rois = torch.cat([fake_inds, bboxes_tensor], dim=1)  # Nx5
    rois = rois.to(device=device)
    if num_bbox > 0:
        gt_masks_th = annos.index_select(0, inds_tensor).to(dtype=rois.dtype)
        targets = roi_align(gt_masks_th[:, None, :, :], rois, out_shape, 1.0, 0, True).squeeze(1)
        resized_masks = targets >= 0.5
    else:
        resized_masks = torch.empty((0, *out_shape), device=device)
    return resized_masks.float()


def gt_instances_to_rle(
    masks: tv_tensors.Mask | None,
    polygons: tv_tensors.KeyPoints | None,
    img_info: ImageInfo | None = None,
) -> list[dict]:
    """Encode one sample's ground-truth instances as RLE, from whichever carrier is populated.

    ``InstanceSegmentationSample`` carries ``masks``; ``InstanceSegmentationPolygonSample``
    carries ``polygons`` and has no ``masks`` column at all. Models normally rasterize
    polygons in ``_customize_inputs``, but ground truth that bypasses the model boundary
    -- tiled validation, and OpenVINO evaluation, which has no ``_customize_inputs`` --
    still arrives unrasterized here.

    Args:
        masks: ``(N, H, W)`` ground-truth masks, or ``None``.
        polygons: ``(N, P_max, 2)`` NaN-padded polygons, or ``None``.
        img_info: Fallback source of the canvas size when ``polygons`` carries none.

    Returns:
        One RLE dict per instance; empty for a no-annotation sample.

    Raises:
        ValueError: If neither carrier is populated, or polygons have no resolvable canvas.
    """
    if masks is not None:
        return [encode_rle(m) for m in masks.data] if len(masks) else []
    if polygons is None:
        msg = "Either masks or polygons are required for metric computation"
        raise ValueError(msg)
    canvas = getattr(polygons, "canvas_size", None)
    if canvas is None:
        if img_info is None:
            msg = "Cannot rasterize polygons without a canvas size or img_info"
            raise ValueError(msg)
        canvas = img_info.ori_shape
    return [encode_rle(m) for m in rasterize(polygons, canvas_size=canvas, dtype=torch.uint8)]
