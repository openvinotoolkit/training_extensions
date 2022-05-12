"""This file defines the ShapeEntity interface and the Shape abstract class"""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import abc
import datetime
import warnings
from enum import IntEnum, auto
from typing import TYPE_CHECKING

from shapely.errors import PredicateError, TopologicalError
from shapely.geometry import Polygon as shapely_polygon

if TYPE_CHECKING:
    from ote_sdk.entities.shapes.rectangle import Rectangle


class GeometryException(ValueError):
    """Exception that is thrown if the geometry of a Shape is invalid"""


class ShapeType(IntEnum):
    """Shows which type of Shape is being used"""

    ELLIPSE = auto()
    RECTANGLE = auto()
    POLYGON = auto()


class ShapeEntity(metaclass=abc.ABCMeta):
    """
    This interface represents the annotation shapes on the media given by user annotations or system analysis.
    The shapes is a 2D geometric shape living in a normalized coordinate system (the values range from 0 to 1).
    """

    # pylint: disable=redefined-builtin
    def __init__(self, shape_type: ShapeType):
        self._type = shape_type

    @property
    def type(self) -> ShapeType:
        """
        Get the type of Shape that this Shape represents
        """
        return self._type

    @abc.abstractmethod
    def get_area(self) -> float:
        """
        Get the area of the shape
        """
        raise NotImplementedError

    @abc.abstractmethod
    def intersects(self, other: "Shape") -> bool:
        """
        Returns true if other intersects with shape, otherwise returns false

        :param other: Shape to compare with

        :return: true if other intersects with shape, otherwise returns false
        """
        raise NotImplementedError

    @abc.abstractmethod
    def contains_center(self, other: "ShapeEntity") -> bool:
        """
        Checks whether the center of the 'other' shape is located in the shape.

        :param other: Shape to compare with
        :return: Boolean that indicates whether the center of the other shape is located in the shape
        """
        raise NotImplementedError

    @abc.abstractmethod
    def normalize_wrt_roi_shape(self, roi_shape: "Rectangle") -> "Shape":
        """
        The inverse of denormalize_wrt_roi_shape.
        Transforming shape from the `roi` coordinate system to the normalized coordinate system.
        This is used when the tasks want to save the analysis results.

        For example in Detection -> Segmentation pipeline, the analysis results of segmentation
        needs to be normalized to the roi (bounding boxes) coming from the detection.

        :param roi_shape:
        """
        raise NotImplementedError

    @abc.abstractmethod
    def denormalize_wrt_roi_shape(self, roi_shape: "Rectangle") -> "Shape":
        """
        The inverse of normalize_wrt_roi_shape.
        Transforming shape from the normalized coordinate system to the `roi` coordinate system.
        This is used to pull ground truth during training process of the tasks.
        Examples given in the Shape implementations.

        :param roi_shape:
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _as_shapely_polygon(self) -> shapely_polygon:
        """
        Convert shape to a shapely polygon. Shapely polygons are within the SDK used to calculate the intersection
        between Shapes. It is also used in the SDK to find shapes that are visible within a given ROI.
        :return: shapely polygon
        """
        raise NotImplementedError


class Shape(ShapeEntity):
    """
    Base class for Shape entities
    """

    # pylint: disable=redefined-builtin, too-many-arguments; Requires refactor
    def __init__(self, shape_type: ShapeType, modification_date: datetime.datetime):
        super().__init__(shape_type=shape_type)
        self.modification_date = modification_date

    def __repr__(self):
        return f"Shape with modification date:('{self.modification_date}')"

    def get_area(self) -> float:
        raise NotImplementedError

    # pylint: disable=protected-access
    def intersects(self, other: "Shape") -> bool:
        polygon_roi = self._as_shapely_polygon()
        polygon_shape = other._as_shapely_polygon()
        try:
            return polygon_roi.intersects(polygon_shape)
        except (PredicateError, TopologicalError) as exception:
            raise GeometryException(
                f"The intersection between the shapes {self} and {other} could not be computed: "
                f"{exception}."
            ) from exception

    # pylint: disable=protected-access
    def contains_center(self, other: "ShapeEntity") -> bool:
        """
        Checks whether the center of the 'other' shape is located in the shape.

        :param other: Shape to compare with
        :return: Boolean that indicates whether the center of the other shape is located in the shape
        """
        polygon_roi = self._as_shapely_polygon()
        polygon_shape = other._as_shapely_polygon()
        return polygon_roi.contains(polygon_shape.centroid)

    def _validate_coordinates(self, x: float, y: float) -> bool:
        """
        Checks whether the values for a given x,y coordinate pair lie within the range of (0,1) that is expected for
        the normalized coordinate system. Issues a warning if the coordinates are out of bounds.

        :param x: x-coordinate to validate
        :param y: y-coordinate to validate

        :return: ``True`` if coordinates are within expected range, ``False`` otherwise
        """
        if not ((0.0 <= x <= 1.0) and (0.0 <= y <= 1.0)):
            warnings.warn(
                f"{type(self).__name__} coordinates (x={x}, y={y}) are out of bounds, a normalized "
                f"coordinate system is assumed. All coordinates are expected to be in range (0,1).",
                UserWarning,
            )
            return False
        return True

    def __hash__(self):
        return hash(str(self))
