"""Convert a mask to a border image."""

# Copyright (C) 2021-2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from __future__ import annotations

import cv2
import numpy as np
from skimage.measure import find_contours, label, regionprops

from otx.v2.api.entities.shapes.polygon import Polygon
from otx.v2.api.entities.utils.shape_factory import ShapeFactory


def mask_to_border(mask: np.ndarray) -> np.ndarray:
    """Make a border by using a binary mask.

    Args:
        mask (np.ndarray): Input binary mask

    Returns:
        np.ndarray: Border image.
    """
    h, w = mask.shape
    border = np.zeros((h, w))

    contours = find_contours(mask, 0.5)  # since the input range is [0, 1], the threshold is 0.5
    for contour in contours:
        for c in contour:
            x = int(c[0])
            y = int(c[1])
            border[x][y] = 1  # since the input is binary, the value is 1

    return border


def mask2bbox(mask: np.ndarray) -> list[list[int]]:
    """Mask to bounding boxes.

    Args:
        mask (np.ndarray): Input binary mask

    Returns:
        List[List[int]]: Bounding box coordinates
    """
    bboxes: list[list[int]] = []

    mask = mask_to_border(mask)
    print(np.unique(mask))
    lbl = label(mask)
    props = regionprops(lbl)
    for prop in props:
        x1 = prop.bbox[1]
        y1 = prop.bbox[0]

        x2 = prop.bbox[3]
        y2 = prop.bbox[2]

        bboxes.append([x1, y1, x2, y2])

    return bboxes


def convert_polygon_to_mask(shape: Polygon, width: int, height: int) -> np.ndarray:
    """Convert polygon to mask.

    Args:
        shape (Polygon): Polygon to convert.
        width (int): Width of image.
        height (int): Height of image.

    Returns:
        np.ndarray: Generated mask from given polygon.
    """
    polygon = ShapeFactory.shape_as_polygon(shape)
    contour = [[int(point.x * width), int(point.y * height)] for point in polygon.points]
    gt_mask = np.zeros(shape=(height, width), dtype=np.uint8)
    return cv2.drawContours(gt_mask, np.asarray([contour]), 0, 1, -1)


def generate_fitted_bbox(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    width: int,
    height: int,
) -> list[int]:
    """Generate bounding fitted box.

    Args:
        x1, y1, x2, y2 (int): Bounding box coordinates.
        width (int): Width of image.
        height (int): Height of image.

    Returns:
        list[int]: Generated bounding box.
    """
    return [
        max(0, x1),
        max(0, y1),
        min(width, x2),
        min(height, y2),
    ]


def generate_bbox_with_perturbation(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    width: int,
    height: int,
    offset_bbox: int = 0,
) -> list[int]:
    """Generate bounding box with perturbation.

    Args:
        x1, y1, x2, y2 (int): Bounding box coordinates.
        width (int): Width of image.
        height (int): Height of image.
        offset_bbox (int): Offset to apply to the bounding box, defaults to 0.

    Returns:
        list[int]: Generated bounding box.
    """

    def get_randomness(length: int) -> int:
        if offset_bbox == 0:
            return 0
        rng = np.random.default_rng()
        return int(rng.normal(0, min(int(length * 0.1), offset_bbox)))

    return generate_fitted_bbox(
        x1 + get_randomness(width),
        y1 + get_randomness(height),
        x2 + get_randomness(width),
        y2 + get_randomness(height),
        width,
        height,
    )


def generate_bbox_from_mask(gt_mask: np.ndarray, width: int, height: int) -> list[int]:
    """Generate bounding box from given mask.

    Args:
        gt_mask (np.ndarry): Mask to generate bounding box.
        width (int): Width of image.
        height (int): Height of image.

    Returns:
        list[int]: Generated bounding box from given mask.
    """
    x_indices: np.ndarray
    y_indices: np.ndarray
    x_min, x_max = 0, width
    y_min, y_max = 0, height
    y_indices, x_indices = np.where(gt_mask == 1)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    return generate_bbox_with_perturbation(x_min, y_min, x_max, y_max, width, height)
