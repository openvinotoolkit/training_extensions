"""This module implements the RotatedRectangle shape entity"""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import datetime
import math
import warnings
from operator import attrgetter
from typing import List, Optional

from shapely.geometry import Polygon as shapely_polygon

from ote_sdk.entities.scored_label import ScoredLabel
from ote_sdk.entities.shapes.polygon import Point
from ote_sdk.entities.shapes.rectangle import Rectangle
from ote_sdk.entities.shapes.shape import Shape, ShapeType
from ote_sdk.utils.time_utils import now


class RotatedRectangle(Shape):
    """
    RotatedRectangle represents a rectangular shape.

    RotatedRectangle is used to localize rotated objects.

    :param points: list of Point's forming the rotated rectangle
    :param labels: list of the ScoredLabel's for the rotated rectangle
    :param modification_date: last modified date
    """

    def __init__(
        self,
        points: List[Point],
        labels: Optional[List[ScoredLabel]] = None,
        modification_date: Optional[datetime.datetime] = None,
    ):
        labels = [] if labels is None else labels
        modification_date = now() if modification_date is None else modification_date
        super().__init__(
            type=ShapeType.ROTATED_RECTANGLE,
            labels=labels,
            modification_date=modification_date,
        )

        if len(points) != 4:
            raise ValueError(
                f"Invalid number of points have been passed. Expected: 4, actual: {len(points)}."
            )

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
        return (
            "RotatedRectangle(points=["
            f"{self.points[0]}, "
            f"{self.points[1]}, "
            f"{self.points[2]}, "
            f"{self.points[3]}])"
        )

    def __eq__(self, other):
        if isinstance(other, RotatedRectangle):
            return (
                self.points == other.points
                and self.modification_date == other.modification_date
            )
        return False

    def __hash__(self):
        return hash(str(self))

    def normalize_wrt_roi_shape(self, roi_shape: Rectangle) -> "RotatedRectangle":
        """
        Transforms from the `roi` coordinate system to the normalized coordinate system.
        This function is the inverse of ``denormalize_wrt_roi_shape``.

        :example: Assume we have RotatedRectangle `p1` which lives in the middle of a 2D space.
            The 2D space where `p1` lives in is an `roi` living in the top-left quarter of the normalized coordinate
            space. This function returns RotatedRectangle `p1` expressed in the normalized coordinate space.

            >>> from ote_sdk.entities.annotation import Annotation
            >>> from ote_sdk.entities.shapes.polygon import Point
            >>> from ote_sdk.entities.shapes.rectangle import Rectangle
            >>> from ote_sdk.entities.shapes.rotated_rectangle import RotatedRectangle
            >>> p1 = RotatedRectangle(points=[Point(x=0.5, y=0.25),
            ... Point(x=0.75, y=0.5), Point(x=0.5, y=0.75), Point(x=0.25, y=0.5)])
            >>> roi = Rectangle(x1=0.0, x2=0.5, y1=0.0, y2=0.5)
            >>> normalized = p1.normalize_wrt_roi_shape(roi)
            >>> normalized
            RotatedRectangle(points=[Point(0.25, 0.125), Point(0.375, 0.25), Point(0.25, 0.375), Point(0.125, 0.25)])

        :param roi_shape: Region of Interest
        :return: New RotatedRectangle in the image coordinate system
        """

        if not isinstance(roi_shape, Rectangle):
            raise ValueError("roi_shape has to be a Rectangle.")

        roi_shape = roi_shape.clip_to_visible_region()

        points = [p.normalize_wrt_roi(roi_shape) for p in self.points]
        return RotatedRectangle(points=points)

    def denormalize_wrt_roi_shape(self, roi_shape: Rectangle) -> "RotatedRectangle":
        """
        Transforming shape from the normalized coordinate system to the `roi` coordinate system.
        This function is the inverse of ``normalize_wrt_roi_shape``

        :example: Assume we have RotatedRectangle `p1` which lives in the middle of the normalized coordinate space.
            The `roi` is a rectangle `roi` living in the top-left quarter of the normalized coordinate
            space.
            This function returns RotatedRectangle `p1`expressed in the coordinate space of `roi`.

            RotatedRectangle denormalized to a rectangle as ROI

        >>> from ote_sdk.entities.annotation import Annotation
        >>> from ote_sdk.entities.shapes.polygon import Point
        >>> from ote_sdk.entities.shapes.rectangle import Rectangle
        >>> from ote_sdk.entities.shapes.rotated_rectangle import RotatedRectangle
        >>> p1 = RotatedRectangle(points=[Point(0.25, 0.125), Point(0.375, 0.25),
        ... Point(0.25, 0.375), Point(0.125, 0.25)])
        >>> roi = Rectangle(x1=0.0, x2=0.5, y1=0.0, y2=0.5)
        >>> normalized = p1.denormalize_wrt_roi_shape(roi)
        >>> normalized
        RotatedRectangle(points=[Point(0.5, 0.25), Point(0.75, 0.5), Point(0.5, 0.75), Point(0.25, 0.5)])

        :param roi_shape: Region of Interest
        :return: New RotatedRectangle in the ROI coordinate system
        """

        if not isinstance(roi_shape, Rectangle):
            raise ValueError("roi_shape has to be a Rectangle.")

        roi_shape = roi_shape.clip_to_visible_region()

        points = [p.denormalize_wrt_roi_shape(roi_shape) for p in self.points]
        return RotatedRectangle(points=points)

    def _as_shapely_polygon(self) -> shapely_polygon:
        """
        Returns the RotatedRectangle object as a shapely polygon
        which is used for calculating intersection between shapes.
        """
        return shapely_polygon([(point.x, point.y) for point in self.points])

    def get_area(self) -> float:
        """
        Returns the approximate area of the shape. Area is a value between 0 and 1,
        computed by converting the RotatedRectangle to a shapely polygon and reading the `.area` property.

        NOTE: This method should not be relied on for exact area computation. The area is approximate, because shapes
        are continuous, but pixels are discrete.

        :example:

            >>> Polygon(points=[Point(x=0.0, y=0.5), Point(x=0.5, y=0.5), Point(x=0.75, y=0.75)]).get_area()
            0.0625

        :return: area of the shape
        """
        return self._as_shapely_polygon().area

    def get_orientation(self) -> "Point":
        """
        Returns a unit vector that defines orientation of RotatedRectangle.
        If RotatedRectangle is defined by four points: p0, p1, p2, p3, then
        its direction matches the direction of (p1 - p2) vector.
        """

        vect = self.points[1].x - self.points[2].x, self.points[1].y - self.points[2].y
        norm = math.sqrt(vect[0] * vect[0] + vect[1] * vect[1])
        if norm == 0:
            raise RuntimeError(
                f"Invalid {self}, it is impossible to define orientation."
            )
        return Point(vect[0] / norm, vect[1] / norm)
