# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""OTX image type definition."""

from __future__ import annotations

from enum import Enum, IntEnum, auto

import numpy as np
from torchvision import tv_tensors


class ImageColorChannel(str, Enum):
    """ImageColorChannel definition."""

    RGB = "RGB"
    BGR = "BGR"


class ImageType(IntEnum):
    """Enum to indicate the image type in `ImageInfo` class."""

    NUMPY = auto()
    TV_IMAGE = auto()
    NUMPY_LIST = auto()
    TV_IMAGE_LIST = auto()

    @classmethod
    def get_image_type(
        cls,
        image: np.ndarray | tv_tensors.Image | list[np.ndarray] | list[tv_tensors.Image],
    ) -> ImageType:
        """Infer the image type from the given image object."""
        if isinstance(image, np.ndarray):
            return ImageType.NUMPY
        if isinstance(image, tv_tensors.Image):
            return ImageType.TV_IMAGE
        if isinstance(image, list):
            image = next(iter(image))
            if isinstance(image, np.ndarray):
                return ImageType.NUMPY_LIST
            if isinstance(image, tv_tensors.Image):
                return ImageType.TV_IMAGE_LIST
        raise TypeError(image)
