# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Utility functions for mask operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pycocotools.mask as mask_utils
import torch

if TYPE_CHECKING:
    import numpy as np
    from datumaro import Polygon


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
    return mask_utils.frPyObjects(polygons, height, width)


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
