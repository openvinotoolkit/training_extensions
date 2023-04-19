"""This module implements the Image entity."""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


from typing import Optional, Tuple, Callable, Union

import cv2
import imagesize
import numpy as np

from otx.api.entities.annotation import Annotation
from otx.api.entities.media import IMedia2DEntity
from otx.api.entities.shapes.rectangle import Rectangle


class Image(IMedia2DEntity):
    """Represents a 2D image.

    The image must be instantiated with either a NumPy array containing the image data
    or a path to an image file.

    Args:
        data (Optional[np.ndarray]): NumPy data.
        file_path (Optional[str]): Path to image file.
    """

    # pylint: disable=too-many-arguments, redefined-builtin
    def __init__(
        self,
        data: Optional[Union[np.ndarray, Callable[[], np.ndarray]]] = None,
        file_path: Optional[str] = None,
        size: Optional[Union[Tuple[int, int], Callable[[], Tuple[int, int]]]] = None,
    ):
        if (data is None) == (file_path is None):
            raise ValueError("Either path to image file or image data should be provided.")
        self.__data: Optional[Union[np.ndarray, Callable[[], np.ndarray]]] = data
        self.__file_path: Optional[str] = file_path
        self.__height: Optional[int] = None
        self.__width: Optional[int] = None
        # TODO: refactor this
        self.__size: Optional[Union[Tuple[int, int], Callable[[], Tuple[int, int]]]] = size

    def __str__(self):
        """String representation of the image. Returns the image format, name and dimensions."""
        return (
            f"{self.__class__.__name__}"
            f"({self.__file_path if self.__data is None else 'with data'}, "
            f"width={self.width}, height={self.height})"
        )

    def __get_size(self) -> Tuple[int, int]:
        """Returns image size.

        Returns:
            Tuple[int, int]: Image size as a (height, width) tuple.
        """
        if callable(self.__size):
            height, width = self.__size()
            self.__size = None
            return height, width
        if self.__size is not None:
            height, width = self.__size
            self.__size = None
            return height, width
        if callable(self.__data):
            height, width = self.__data().shape[:2]
            return height, width
        if self.__data is not None:
            return self.__data.shape[0], self.__data.shape[1]
        if self.__file_path is not None:
            try:
                width, height = imagesize.get(self.__file_path)
                if width <= 0 or height <= 0:
                    raise ValueError("Invalide image size")
            except ValueError:
                image = cv2.imread(self.__file_path)
                height, width = image.shape[:2]
            return height, width
        raise NotImplementedError

    @property
    def numpy(self) -> np.ndarray:
        """Numpy representation of the image.

        For color images the dimensions are (height, width, color) with RGB color channel order.

        Returns:
            np.ndarray: NumPy representation of the image.
        """
        if self.__data is None:
            return cv2.cvtColor(cv2.imread(self.__file_path), cv2.COLOR_BGR2RGB)
        if callable(self.__data):
            return self.__data()
        return self.__data

    @numpy.setter
    def numpy(self, value: np.ndarray):
        self.__data = value
        self.__file_path = None
        self.__size = None
        self.__height, self.__width = self.__get_size()

    def roi_numpy(self, roi: Optional[Annotation] = None) -> np.ndarray:
        """Obtains the numpy representation of the image for a selection region of interest (roi).

        Args:
            roi (Optional[Annotaiton]): The region of interest can be Rectangle in the relative
                coordinate system of the full-annotation.

        Returns:
            np.ndarray: Selected region as numpy array.
        """
        data = self.numpy
        if roi is None:
            return data

        if not isinstance(roi.shape, Rectangle):
            raise ValueError("roi shape is not a Rectangle")

        if data is None:
            raise ValueError("Numpy array is None, and thus cannot be cropped")

        if len(data.shape) < 2:
            raise ValueError("This image is one dimensional, and thus cannot be cropped")

        return roi.shape.crop_numpy_array(data)

    @property
    def height(self) -> int:
        """Returns the height of the image."""
        if self.__height is None:
            self.__height, self.__width = self.__get_size()
        return self.__height

    @property
    def width(self) -> int:
        """Returns the width of the image."""
        if self.__width is None:
            self.__height, self.__width = self.__get_size()
        return self.__width

    @property
    def path(self) -> Optional[str]:
        """Returns the file path of the image."""
        return self.__file_path
