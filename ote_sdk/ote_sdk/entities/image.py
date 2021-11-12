"""This module implements the Image entity."""

# INTEL CONFIDENTIAL
#
# Copyright (C) 2021 Intel Corporation
#
# This software and the related documents are Intel copyrighted materials, and
# your use of them is governed by the express license under which they were provided to
# you ("License"). Unless the License provides otherwise, you may not use, modify, copy,
# publish, distribute, disclose or transmit this software or the related documents
# without Intel's prior written permission.
#
# This software and the related documents are provided as is,
# with no express or implied warranties, other than those that are expressly stated
# in the License.


from typing import Optional, Tuple

import cv2
import numpy as np

from ote_sdk.entities.annotation import Annotation
from ote_sdk.entities.media import IMedia2DEntity
from ote_sdk.entities.shapes.rectangle import Rectangle


class Image(IMedia2DEntity):
    """
    Represents a 2D image.

    The image must be instantiated with either a NumPy array containing the image data
    or a path to an image file.

    :param data: NumPy data.
    :param file_path: Path to image file.
    """

    # pylint: disable=too-many-arguments, redefined-builtin
    def __init__(
        self,
        data: Optional[np.ndarray] = None,
        file_path: Optional[str] = None,
    ):
        if (data is None) == (file_path is None):
            raise ValueError(
                "Either path to image file or image data should be provided."
            )
        self.__data: Optional[np.ndarray] = data
        self.__file_path: Optional[str] = file_path
        self.__height: Optional[int] = None
        self.__width: Optional[int] = None

    def __str__(self):
        return (
            f"{self.__class__.__name__}"
            f"({self.__file_path if self.__data is None else 'with data'}, "
            f"width={self.width}, height={self.height})"
        )

    def __get_size(self) -> Tuple[int, int]:
        """
        Returns image size.

        :return: Image size as a (height, width) tuple.
        """
        if self.__data is not None:
            return self.__data.shape[0], self.__data.shape[1]
        # TODO(pdruzhkov). Get image size w/o reading & decoding its data.
        image = cv2.imread(self.__file_path)
        return image.shape[:2]

    @property
    def numpy(self) -> np.ndarray:
        """
        NumPy representation of the image.

        For color images the dimensions are (height, width, color) with RGB color channel order.
        """
        if self.__data is None:
            return cv2.cvtColor(cv2.imread(self.__file_path), cv2.COLOR_BGR2RGB)
        return self.__data

    @numpy.setter
    def numpy(self, value: np.ndarray):
        self.__data = value
        self.__file_path = None
        self.__height, self.__width = self.__get_size()

    def roi_numpy(self, roi: Optional[Annotation] = None) -> np.ndarray:
        """
        Obtains the numpy representation of the image for a selection region of interest (roi).

        :param roi: The region of interest can be Rectangle in the relative coordinate system of the full-annotation.
        :return: Selected region as numpy array.
        """
        data = self.numpy
        if roi is None:
            return data

        if not isinstance(roi.shape, Rectangle):
            raise ValueError("roi shape is not a Rectangle")

        if data is None:
            raise ValueError("Numpy array is None, and thus cannot be cropped")

        if len(data.shape) < 2:
            raise ValueError(
                "This image is one dimensional, and thus cannot be cropped"
            )

        return roi.shape.crop_numpy_array(data)

    @property
    def height(self) -> int:
        """
        Returns the height of the image.
        """
        if self.__height is None:
            self.__height, self.__width = self.__get_size()
        return self.__height

    @property
    def width(self) -> int:
        """
        Returns the width of the image.
        """
        if self.__width is None:
            self.__height, self.__width = self.__get_size()
        return self.__width
