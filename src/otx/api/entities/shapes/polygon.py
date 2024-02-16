"""This module implements the Polygon Shape entity."""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# Conflict with Isort
# pylint: disable=wrong-import-order

import datetime
import warnings
from operator import attrgetter
from typing import List, Optional
import numpy as np

from shapely.geometry import Polygon as shapely_polygon

from otx.api.entities.shapes.rectangle import Rectangle
from otx.api.entities.shapes.shape import Shape, ShapeType
from otx.api.utils.time_utils import now


class Point:
    """This class defines a Point with an X and Y coordinate.

    Multiple points can be used to represent a Polygon
    """

    __slots__ = ["x", "y"]

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __repr__(self):
        """String representation of the point."""
        return f"Point({self.x}, {self.y})"

    def __eq__(self, other):
        """Checks if two points have the same x and y coordinates."""
        if isinstance(other, Point):
            return self.x == other.x and self.y == other.y
        return False

    def normalize_wrt_roi(self, roi_shape: Rectangle) -> "Point":
        """The inverse of denormalize_wrt_roi_shape.

        Transforming Polygon from the `roi` coordinate system to the normalized coordinate system.
        This is used when the tasks want to save the analysis results.

        For example in Detection -> Segmentation pipeline, the analysis results of segmentation
        needs to be normalized to the roi (bounding boxes) coming from the detection.

        Args:
            roi_shape (Point): the shape of the roi
        """
        roi_shape = roi_shape.clip_to_visible_region()
        width = roi_shape.width
        height = roi_shape.height
        x1 = roi_shape.x1
        y1 = roi_shape.y1
        return Point(x=self.x * width + x1, y=self.y * height + y1)

    def denormalize_wrt_roi_shape(self, roi_shape: Rectangle) -> "Point":
        """The inverse of normalize_wrt_roi_shape.

        Transforming Polygon from the normalized coordinate system to the `roi` coordinate system.
        This is used to pull ground truth during training process of the tasks.
        Examples given in the Shape implementations.

        Args:
            roi_shape (Rectangle): the shape of the roi
        """
        roi_shape = roi_shape.clip_to_visible_region()

        return Point(
            x=(self.x - roi_shape.x1) / roi_shape.width,
            y=(self.y - roi_shape.y1) / roi_shape.height,
        )


class Polygon(Shape):
    """Represents a polygon formed by a list of coordinates.

    NB Freehand drawings are also stored as polygons.

    Args:
        points: list of Point's forming the polygon
        modification_date: last modified date
    """

    # pylint: disable=too-many-arguments; Requires refactor
    def __init__(
        self,
        points: List[Point],
        modification_date: Optional[datetime.datetime] = None,
    ):
        modification_date = now() if modification_date is None else modification_date
        super().__init__(
            shape_type=ShapeType.POLYGON,
            modification_date=modification_date,
        )

        if len(points) == 0:
            raise ValueError("Cannot create polygon with no points")

        self.points = points

        self.min_x = min(points, key=attrgetter("x")).x
        self.max_x = max(points, key=attrgetter("x")).x
        self.min_y = min(points, key=attrgetter("y")).y
        self.max_y = max(points, key=attrgetter("y")).y

        is_valid = True
        for (x, y) in [(self.min_x, self.min_y), (self.max_x, self.max_y)]:
            is_valid = is_valid and self._validate_coordinates(x, y)
        if not is_valid:
            points_str = "; ".join(str(p) for p in self.points)
            warnings.warn(
                f"{type(self).__name__} coordinates are invalid : {points_str}",
                UserWarning,
            )

    def __repr__(self):
        """String representation of the polygon."""
        return (
            f"Polygon(len(points)={len(self.points)},"
            f" min_x={self.min_x}, max_x={self.max_x}, min_y={self.min_y}, max_y={self.max_y})"
        )

    def __eq__(self, other):
        """Compares if the polygon has the same points and modification date."""
        if isinstance(other, Polygon):
            return self.points == other.points and self.modification_date == other.modification_date
        return False

    def __hash__(self):
        """Returns hash of the Polygon object."""
        return hash(str(self))

    def normalize_wrt_roi_shape(self, roi_shape: Rectangle) -> "Polygon":
        """Transforms from the `roi` coordinate system to the normalized coordinate system.

        This function is the inverse of ``denormalize_wrt_roi_shape``.

        Example:
            Assume we have Polygon `p1` which lives in the top-right quarter of a 2D space.
            The 2D space where `p1` lives in is an `roi` living in the top-left quarter of the normalized coordinate
            space. This function returns Polygon `p1` expressed in the normalized coordinate space.

            >>> from otx.api.entities.annotation import Annotation
            >>> from otx.api.entities.shapes.rectangle import Rectangle
            >>> p1 = Polygon(points=[Point(x=0.5, y=0.0), Point(x=0.75, y=0.2), Point(x=0.6, y=0.1)])
            >>> roi = Rectangle(x1=0.0, x2=0.5, y1=0.0, y2=0.5)
            >>> normalized = p1.normalize_wrt_roi_shape(roi_shape)
            >>> normalized
            Polygon(, len(points)=3)

        Args:
            roi_shape: Region of Interest

        Returns:
            New polygon in the image coordinate system
        """
        if not isinstance(roi_shape, Rectangle):
            raise ValueError("roi_shape has to be a Rectangle.")

        roi_shape = roi_shape.clip_to_visible_region()

        points = [p.normalize_wrt_roi(roi_shape) for p in self.points]
        return Polygon(points=points)

    def denormalize_wrt_roi_shape(self, roi_shape: Rectangle) -> "Polygon":
        """Transforming shape from the normalized coordinate system to the `roi` coordinate system.

        This function is the inverse of ``normalize_wrt_roi_shape``

        Example:
            Assume we have Polygon `p1` which lives in the top-right quarter of the normalized coordinate space.
            The `roi` is a rectangle living in the half right of the normalized coordinate space.
            This function returns Polygon `p1` expressed in the coordinate space of `roi`. (should return top-half)

            Polygon denormalized to a rectangle as ROI

            >>> from otx.api.entities.shapes.rectangle import Rectangle
            >>> from otx.api.entities.annotation import Annotation
            >>> p1 = Polygon(points=[Point(x=0.5, y=0.0), Point(x=0.75, y=0.2), Point(x=0.6, y=0.1)])
            >>> roi = Rectangle(x1=0.5, x2=1.0, y1=0.0, y2=1.0)  # the half-right
            >>> normalized = p1.denormalize_wrt_roi_shape(roi_shape)
            >>> normalized
            Polygon(, len(points)=3)

        Args:
            roi_shape: Region of Interest

        Returns:
            New polygon in the ROI coordinate system
        """
        if not isinstance(roi_shape, Rectangle):
            raise ValueError("roi_shape has to be a Rectangle.")

        roi_shape = roi_shape.clip_to_visible_region()

        points = [p.denormalize_wrt_roi_shape(roi_shape) for p in self.points]
        return Polygon(points=points)

    def _as_shapely_polygon(self) -> shapely_polygon:
        """Returns the Polygon object as a shapely polygon which is used for calculating intersection between shapes."""
        return shapely_polygon([(point.x, point.y) for point in self.points])

    def get_area(self) -> float:
        """Returns the approximate area of the shape.

        Area is a value between 0 and 1, computed by converting the Polygon to a shapely polygon and reading the
        `.area` property.

        NOTE: This method should not be relied on for exact area computation. The area is approximate, because shapes
        are continuous, but pixels are discrete.

        Example:

            >>> Polygon(points=[Point(x=0.0, y=0.5), Point(x=0.5, y=0.5), Point(x=0.75, y=0.75)]).get_area()
            0.0625

        Returns:
            area of the shape
        """
        return self._as_shapely_polygon().area
