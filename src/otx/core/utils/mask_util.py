# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Utility functions for mask operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pycocotools.mask as mask_utils

if TYPE_CHECKING:
    import numpy as np
    from datumaro import Polygon


def polygon_to_bitmap(polygons: list[Polygon], height: int, width: int) -> np.ndarray:
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
