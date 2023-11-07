"""Collection of dataset I/O functions."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
from torchvision.datasets.folder import IMG_EXTENSIONS

if TYPE_CHECKING:
    import numpy as np


def get_image_filenames(path: str | Path) -> list[Path]:
    """Get image filenames.

    Args:
        path (str | Path): Path to image or image-folder.

    Returns:
        list[Path]: List of image filenames

    """
    image_filenames: list[Path] = []

    if isinstance(path, str):
        path = Path(path)

    if path.is_file() and path.suffix.lower() in IMG_EXTENSIONS:
        image_filenames = [path]

    if path.is_dir():
        image_filenames = [p for p in path.glob("**/*") if p.suffix.lower() in IMG_EXTENSIONS]

    if not image_filenames:
        msg = f"Found 0 images in {path}"
        raise ValueError(msg)

    return image_filenames


def get_image_height_and_width(image_size: int | tuple[int, int]) -> tuple[int, int]:
    """Get image height and width from ``image_size`` variable.

    Args:
        image_size (int | tuple[int, int] | None, optional): Input image size.

    Raises:
        ValueError: Image size not None, int or tuple.

    Examples:
        >>> get_image_height_and_width(image_size=256)
        (256, 256)

        >>> get_image_height_and_width(image_size=(256, 256))
        (256, 256)

        >>> get_image_height_and_width(image_size=(256, 256, 3))
        (256, 256)

        >>> get_image_height_and_width(image_size=256.)
        Traceback (most recent call last):
        File "<string>", line 1, in <module>
        File "<string>", line 18, in get_image_height_and_width
        TypeError: ``image_size`` could be either int or tuple[int, int]

    Returns:
        tuple[int | None, int | None]: A tuple containing image height and width values.
    """
    if isinstance(image_size, int):
        height_and_width = (image_size, image_size)
    elif isinstance(image_size, tuple):
        height_and_width = int(image_size[0]), int(image_size[1])
    else:
        msg = "``image_size`` could be either int or tuple[int, int]"
        raise TypeError(msg)

    return height_and_width


def read_image(path: str | Path, image_size: int | tuple[int, int] | None = None) -> np.ndarray:
    """Read image from disk in RGB format.

    Args:
        path (str, Path): path to the image file

    Example:
        >>> image = read_image("test_image.jpg")

    Returns:
        image as numpy array
    """
    path = path if isinstance(path, str) else str(path)
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if image_size:
        # This part is optional, where the user wants to quickly resize the image
        # with a one-liner code. This would particularly be useful especially when
        # prototyping new ideas.
        height, width = get_image_height_and_width(image_size)
        image = cv2.resize(image, dsize=(width, height), interpolation=cv2.INTER_AREA)

    return image
