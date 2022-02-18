"""This module implements the RotatedRectangle shape entity"""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import datetime
from typing import List, Optional

from ote_sdk.entities.scored_label import ScoredLabel
from ote_sdk.entities.shapes.polygon import Point, Polygon
from ote_sdk.entities.shapes.rectangle import Rectangle


class RotatedRectangle(Polygon):
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
        super().__init__(points, labels, modification_date)

        if len(points) != 4:
            raise ValueError(
                f"Invalid number of points have been passed. Expected: 4, actual: {len(points)}."
            )

    def __repr__(self):
        return (
            "RotatedRectangle(points=["
            f"{self.points[0]}, "
            f"{self.points[1]}, "
            f"{self.points[2]}, "
            f"{self.points[3]}])"
        )

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

        polygon = super().normalize_wrt_roi_shape(roi_shape)
        return RotatedRectangle(
            polygon.points,
            polygon.get_labels(include_empty=True),
            polygon.modification_date,
        )

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

        polygon = super().denormalize_wrt_roi_shape(roi_shape)
        return RotatedRectangle(
            polygon.points,
            polygon.get_labels(include_empty=True),
            polygon.modification_date,
        )
