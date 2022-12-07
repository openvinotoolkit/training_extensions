"""Utils common to converting annotations."""

# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any, Dict, List

import cv2


def create_bboxes_from_mask(mask_path: str) -> List[List[float]]:
    """Create bounding box from binary mask.

    Args:
        mask_path (str): Path to binary mask.

    Returns:
        List[List[float]]: Bounding box coordinates.
    """
    # pylint: disable-msg=too-many-locals

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    height, width = mask.shape

    bboxes: List[List[float]] = []
    _, _, coordinates, _ = cv2.connectedComponentsWithStats(mask)
    for i, coordinate in enumerate(coordinates):
        # First row of the coordinates is always backround,
        # so should be ignored.
        if i == 0:
            continue

        # Last column of the coordinates is the area of the connected component.
        # It could therefore be ignored.
        comp_x, comp_y, comp_w, comp_h, _ = coordinate
        x1 = comp_x / width
        y1 = comp_y / height
        x2 = (comp_x + comp_w) / width
        y2 = (comp_y + comp_h) / height

        bboxes.append([x1, y1, x2, y2])

    return bboxes


def create_polygons_from_mask(mask_path: str) -> List[List[List[float]]]:
    """Create polygons from binary mask.

    Args:
        mask_path (str): Path to binary mask.

    Returns:
        List[List[float]]: Polygon coordinates.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    height, width = mask.shape

    polygons = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    polygons = [[[point[0][0] / width, point[0][1] / height] for point in polygon] for polygon in polygons]

    return polygons


def save_json_items(json_items: Dict[str, Any], file: str) -> None:
    """Save JSON items to file.

    Args:
        json_items (Dict[str, Any]): MVTec AD JSON items
        file (str): Path to save as a JSON file.
    """
    with open(file=file, mode="w", encoding="utf-8") as f:
        json.dump(json_items, f)
