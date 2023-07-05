"""This module implements the Ellipse shape entity."""
# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# Conflict with Isort
# pylint: disable=wrong-import-order

import datetime
import math
from typing import List, Optional, Tuple

import numpy as np
from scipy import optimize, special
from shapely.geometry import Polygon as shapely_polygon

from otx.api.entities.shapes.rectangle import Rectangle
from otx.api.entities.shapes.shape import Shape, ShapeType
from otx.api.utils.time_utils import now

# pylint: disable=invalid-name


class Ellipse(Shape):
    """Ellipse represents an ellipse that is encapsulated by a Rectangle.

    - x1 and y1 represent the top-left coordinate of the encapsulating rectangle
    - x2 and y2 representing the bottom-right coordinate of the encapsulating rectangle

    Args:
        x1: left x coordinate of encapsulating rectangle
        y1: top y coordinate of encapsulating rectangle
        x2: right x coordinate of encapsulating rectangle
        y2: bottom y coordinate of encapsulating rectangle
            modification_date: last modified date
    """

    # pylint: disable=too-many-arguments; Requires refactor
    def __init__(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        modification_date: Optional[datetime.datetime] = None,
    ):
        modification_date = now() if modification_date is None else modification_date
        super().__init__(
            shape_type=ShapeType.ELLIPSE,
            modification_date=modification_date,
        )

        for (x, y) in [(x1, y1), (x2, y2)]:
            self._validate_coordinates(x, y)

        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"Invalid Ellipse with coordinates: x1={self.x1}, y1={self.y1}, x2={self.x2}," f" y2={self.y2}"
            )

    def __repr__(self):
        """Returns the representation of the Ellipse."""
        return f"Ellipse(x1={self.x1}, y1={self.y1}, x2={self.x2}, y2={self.y2})"

    def __eq__(self, other):
        """Returns True if Ellipse is equal to other."""
        if isinstance(other, Ellipse):
            return (
                self.x1 == other.x1
                and self.y1 == other.y1
                and self.x2 == other.x2
                and self.y2 == other.y2
                and self.modification_date == other.modification_date
            )
        return False

    def __hash__(self):
        """Returns the hash of the Ellipse."""
        return hash(str(self))

    @property
    def width(self) -> float:
        """Returns the width [x-axis] of the ellipse.

        Example:

        >>> e1 = Ellipse(x1=0.5, x2=1.0, y1=0.0, y2=0.5)
        >>> e1.width
        0.5

        Returns:
            the width of the ellipse. (x-axis)
        """
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Returns the height [y-axis] of the ellipse.

        Example:

        >>> e1 = Ellipse(x1=0.5, x2=1.0, y1=0.0, y2=0.5)
        >>> e1.height
        0.5

        Returns:
            the height of the ellipse. (y-axis)
        """
        return self.y2 - self.y1

    @property
    def x_center(self) -> float:
        """Returns the x coordinate in the center of the ellipse."""
        return self.x1 + self.width / 2

    @property
    def y_center(self) -> float:
        """Returns the y coordinate in the center of the ellipse."""
        return self.y1 + self.height / 2

    @property
    def minor_axis(self) -> float:
        """Returns the minor axis of the ellipse.

        Example:

        >>> e1 = Ellipse(x1=0.5, x2=1.0, y1=0.0, y2=0.4)
        >>> e1.minor_axis
        0.2

        Returns:
            minor axis of ellipse.
        """
        if self.width > self.height:
            return self.height / 2
        return self.width / 2

    @property
    def major_axis(self) -> float:
        """Returns the major axis of the ellipse.

        Example:

        >>> e1 = Ellipse(x1=0.5, x2=1.0, y1=0.0, y2=0.4)
        >>> e1.major_axis
        0.25

        Returns:
            major axis of ellipse.
        """
        if self.height > self.width:
            return self.height / 2
        return self.width / 2

    def normalize_wrt_roi_shape(self, roi_shape: Rectangle) -> "Ellipse":
        """Transforms from the `roi` coordinate system to the normalized coordinate system.

        This function is the inverse of ``denormalize_wrt_roi_shape``.

        Example:
            Assume we have Ellipse `c1` which lives in the top-right quarter of a 2D space.
            The 2D space where `c1` lives in is an `roi` living in the top-left quarter of the normalized coordinate
            space. This function returns Ellipse `c1` expressed in the normalized coordinate space.

            >>> from otx.api.entities.annotation import Annotation
            >>> from otx.api.entities.shapes.rectangle import Rectangle
            >>> from otx.api.entities.shapes.ellipse import Ellipse
            >>> c1 = Ellipse(x1=0.5, y1=0.5, x2=0.6, y2=0.6)
            >>> roi = Rectangle(x1=0.0, x2=0.5, y1=0.0, y2=0.5)
            >>> normalized = c1.normalize_wrt_roi_shape(roi_shape)
            >>> normalized
            Ellipse(, x1=0.25, y1=0.25, x2=0.3, y2=0.3)

        Args:
            roi_shape: Region of Interest

        Returns:
            New polygon in the image coordinate system
        """

        if not isinstance(roi_shape, Rectangle):
            raise ValueError("roi_shape has to be a Rectangle.")

        roi_shape = roi_shape.clip_to_visible_region()

        return Ellipse(
            x1=self.x1 * roi_shape.width + roi_shape.x1,
            y1=self.y1 * roi_shape.height + roi_shape.y1,
            x2=self.x2 * roi_shape.width + roi_shape.x1,
            y2=self.y2 * roi_shape.height + roi_shape.y1,
        )

    def denormalize_wrt_roi_shape(self, roi_shape: Rectangle) -> "Ellipse":
        """Transforming shape from the normalized coordinate system to the `roi` coordinate system.

        This function is the inverse of ``normalize_wrt_roi_shape``

        Example:
            Assume we have Ellipse `c1` which lives in the top-right quarter of the normalized coordinate space.
            The `roi` is a rectangle living in the half right of the normalized coordinate space.
            This function returns Ellipse `c1` expressed in the coordinate space of `roi`. (should return top-half)

            Ellipse denormalized to a rectangle as ROI

            >>> from otx.api.entities.annotation import Annotation
            >>> from otx.api.entities.shapes.ellipse import Ellipse
            >>> c1 = Ellipse(x1=0.5, x2=1.0, y1=0.0, y2=0.5)  # An ellipse in the top right
            >>> roi = Rectangle(x1=0.5, x2=1.0, y1=0.0, y2=1.0)  # the half-right
            >>> normalized = c1.denormalize_wrt_roi_shape(roi_shape)  # should return top half
            >>> normalized
            Ellipse(, x1=0.0, y1=0.0, x2=1.0, y2=0.5)

        Args:
            roi_shape: Region of Interest

        Returns:
            New polygon in the ROI coordinate system
        """
        if not isinstance(roi_shape, Rectangle):
            raise ValueError("roi_shape has to be a Rectangle.")

        roi_shape = roi_shape.clip_to_visible_region()

        x1 = (self.x1 - roi_shape.x1) / roi_shape.width
        y1 = (self.y1 - roi_shape.y1) / roi_shape.height
        x2 = (self.x2 - roi_shape.x1) / roi_shape.width
        y2 = (self.y2 - roi_shape.y1) / roi_shape.height

        return Ellipse(x1=x1, y1=y1, x2=x2, y2=y2)

    # pylint: disable=no-member; PyLint cannot find scipy.special.ellipeinc()
    def get_evenly_distributed_ellipse_coordinates(self, number_of_coordinates: int = 50) -> List[Tuple[float, float]]:
        """Returns evenly distributed coordinates along the ellipse.

        Makes use of scipy.special.ellipeinc() which provides the numerical integral along the perimeter of the ellipse,
        and scipy.optimize.root() for solving the equal-arcs length equation for the angles.

        Args:
            number_of_coordinates: number of evenly distributed points
                to generate along the ellipsis line

        Returns:
            list of tuple's with coordinates along the ellipse line
        """
        angles = 2 * np.pi * np.arange(number_of_coordinates) / number_of_coordinates
        e = (1.0 - self.minor_axis**2.0 / self.major_axis**2.0) ** 0.5
        total_size = special.ellipeinc(2.0 * np.pi, e)
        arc_size = total_size / number_of_coordinates
        arcs = np.arange(number_of_coordinates) * arc_size
        res = optimize.root(lambda x: (special.ellipeinc(x, e) - arcs), angles)
        angles = res.x
        if self.width > self.height:
            x_points = list(self.major_axis * np.sin(angles))
            y_points = list(self.minor_axis * np.cos(angles))
        else:
            x_points = list(self.minor_axis * np.cos(angles))
            y_points = list(self.major_axis * np.sin(angles))
        coordinates = [
            (point_x + self.x_center, point_y + self.y_center) for point_x, point_y in zip(x_points, y_points)
        ]
        return coordinates

    def _as_shapely_polygon(self) -> shapely_polygon:
        coordinates = self.get_evenly_distributed_ellipse_coordinates()
        return shapely_polygon(coordinates)

    def get_area(self) -> float:
        """Computes the approximate area of the Ellipse.

        Area is a value between 0 and 1, computed as
        `pi * vertex * co-vertex`.

            >>> Ellipse(x1=0, y1=0, x2=0.8, y2=0.4).get_area()
            0.25132741228718347

        Returns:
            area of the shape
        """
        return math.pi * self.minor_axis * self.major_axis
