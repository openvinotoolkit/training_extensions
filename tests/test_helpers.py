# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Helper functions for tests."""

from pathlib import Path

import numpy as np


def generate_random_bboxes(
    image_width: int,
    image_height: int,
    num_boxes: int,
    min_width: int = 10,
    min_height: int = 10,
) -> np.ndarray:
    """Generate random bounding boxes.
    Parameters:
        image_width (int): Width of the image.
        image_height (int): Height of the image.
        num_boxes (int): Number of bounding boxes to generate.
        min_width (int): Minimum width of the bounding box. Default is 10.
        min_height (int): Minimum height of the bounding box. Default is 10.
    Returns:
        ndarray: A NumPy array of shape (num_boxes, 4) representing bounding boxes in format (x_min, y_min, x_max, y_max).
    """
    max_width = image_width - min_width
    max_height = image_height - min_height

    bg = np.random.MT19937(seed=42)
    rg = np.random.Generator(bg)

    x_min = rg.integers(0, max_width, size=num_boxes)
    y_min = rg.integers(0, max_height, size=num_boxes)
    x_max = x_min + rg.integers(min_width, image_width, size=num_boxes)
    y_max = y_min + rg.integers(min_height, image_height, size=num_boxes)

    x_max[x_max > image_width] = image_width
    y_max[y_max > image_height] = image_height
    areas = (x_max - x_min) * (y_max - y_min)
    bboxes = np.column_stack((x_min, y_min, x_max, y_max))
    return bboxes[areas > 0]


def find_folder(base_path: Path, folder_name: str) -> Path:
    """
    Find the folder with the given name within the specified base path.

    Args:
        base_path (Path): The base path to search within.
        folder_name (str): The name of the folder to find.

    Returns:
        Path: The path to the folder.
    """
    for folder_path in base_path.rglob(folder_name):
        if folder_path.is_dir():
            return folder_path
    msg = f"Folder {folder_name} not found in {base_path}."
    raise FileNotFoundError(msg)
