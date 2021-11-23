"""
This module implements helpers for converting shape entities
"""

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

from ote_sdk.entities.shapes.ellipse import Ellipse
from ote_sdk.entities.shapes.polygon import Point, Polygon
from ote_sdk.entities.shapes.rectangle import Rectangle
from ote_sdk.entities.shapes.shape import ShapeEntity


class ShapeFactory:
    """
    Helper class converting between shape types
    """

    @staticmethod
    def shape_as_rectangle(shape: ShapeEntity) -> Rectangle:
        """
        Get the outer-fitted rectangle representation of the shape.
        `media_width` and `media_height` are the width and height of the media in
        which the shape is expressed.

        :example: Let's assume a DatasetItem `dataset_item`.

        To obtain the shapes inside the full annotation as rectangles, one could call:

        >>> from ote_sdk.entities.dataset_item import DatasetItem
        >>> from ote_sdk.entities.image import NullImage
        >>> from ote_sdk.entities.annotation_scene import NullAnnotationScene
        >>> dataset_item = DatasetItem(media=NullImage(),
        >>>     annotation=NullAnnotationScene())
        >>> rectangles = [ShapeFactory.shape_as_rectangle(shape,
        >>>     dataset_item.media.width, dataset_item.media.height) for shape
        ... in dataset_item.annotation_scene.shapes]

        To obtain the shapes inside the dataset item (note that dataset item can have
        roi), one should call:

        >>> rectangles = [ShapeFactory.shape_as_rectangle(shape) for shape in
        >>>     dataset_item.get_annotations()]

        Since the shapes in the first call come from annotation directly, this means
        they are expressed in the media coordinate system. Therefore,
        dataset_item.media.width and dataset_item.media.height are passed.
        While in the second call, the shapes come from denormalization results wrt.
        dataset_item.roi and therefore expressed inside the roi. In this case,
        dataset_item.width and dataset_item.height are passed.

        :example: Converting Ellipse to rectangle

        >>> height = 240
        >>> width = 480
        >>> rectangle = Rectangle(x1=0.375, y1=0.25, x2=0.625, y2=0.75,
        ... labels=[]) # a square of 120 x 120 pixels at the center of the image
        >>> ellipse = Ellipse(x1=0.5, y1=0.5, x2=0.625, y2=0.56125,
        ... labels=[]) # an ellipse of radius 60 pixels (x2 is wrt width) at the
        center of the image
        >>> ellipse_as_rectangle = ShapeFactory.shape_as_rectangle(ellipse)  # get the
        fitted rectangle for the ellipse
        >>> str(rectangle) == str(ellipse_as_rectangle)
        True

        Converting triangle to rectangle

        >>> points = [Point(x=0.5, y=0.25), Point(x=0.375, y=0.75),
        >>>     Point(x=0.625, y=0.75)]
        >>> triangle = Polygon(points=points, labels=[])
        >>> triangle_as_rectangle = ShapeFactory.shape_as_rectangle(triangle)
        >>> str(triangle_as_rectangle) == str(rectangle)
        True

        :param shape: the shape to convert to rectangle
        :return: bounding box of the shape
        """
        if isinstance(shape, Rectangle):
            return shape

        if isinstance(shape, Ellipse):
            x1 = shape.x1
            y1 = shape.y1
            x2 = shape.x2
            y2 = shape.y2
        elif isinstance(shape, Polygon):
            x1 = shape.min_x
            x2 = shape.max_x
            y1 = shape.min_y
            y2 = shape.max_y
        else:
            raise NotImplementedError(f"Conversion of a {type(shape)} to a rectangle is not implemented yet: {shape}")

        new_shape = Rectangle(x1=x1, y1=y1, x2=x2, y2=y2)
        return new_shape

    @staticmethod
    def shape_as_polygon(shape: ShapeEntity) -> Polygon:
        """
        Return a shape converted as polygon. For a rectangle, a polygon will be
        constructed with a point in each corner. For a ellipse, 360 points will be made.
        Otherwise, the original shape will be returned. The width/height for the parent
        need to be specified to make sure the aspect ratio is maintained.

        :param shape: Shape to convert to polygon
        :return: Polygon
        """
        if isinstance(shape, Polygon):
            new_shape = shape
        elif isinstance(shape, Rectangle):
            points = [
                Point(x=shape.x1, y=shape.y1),
                Point(x=shape.x2, y=shape.y1),
                Point(x=shape.x2, y=shape.y2),
                Point(x=shape.x1, y=shape.y2),
                Point(x=shape.x1, y=shape.y1),
            ]
            new_shape = Polygon(points=points)
        elif isinstance(shape, Ellipse):
            coordinates = shape.get_evenly_distributed_ellipse_coordinates()
            points = [Point(x=point[0], y=point[1]) for point in coordinates]
            new_shape = Polygon(points=points)
        else:
            raise NotImplementedError(f"Conversion of a {type(shape)} to a polygon is not implemented yet: " f"{shape}")
        return new_shape

    @staticmethod
    def shape_as_ellipse(shape: ShapeEntity) -> Ellipse:
        """
        Returns the inner-fitted ellipse for a given shape.

        :param shape: Shape to convert
        :return: Converted Shape
        """
        if isinstance(shape, Ellipse):
            return shape

        if isinstance(shape, Rectangle):
            x1 = shape.x1
            x2 = shape.x2
            y1 = shape.y1
            y2 = shape.y2
        elif isinstance(shape, Polygon):
            x1 = shape.min_x
            x2 = shape.max_x
            y1 = shape.min_y
            y2 = shape.max_y
        else:
            raise NotImplementedError(f"Conversion of a {type(shape)} to an ellipse is not implemented yet: {shape}")
        return Ellipse(x1=x1, y1=y1, x2=x2, y2=y2)

    @staticmethod
    def shape_produces_valid_crop(shape: ShapeEntity, media_width: int, media_height: int) -> bool:
        """
        Checks if the shape produces a valid crop based on the image width and height,
        regardless of the contents of the
            image.

        :param shape: the shape to validate
        :param media_width: width of the iamge
        :param media_height height of the image
        :return: whether the shape will produce a valid crop
        """
        is_valid = True
        try:
            shape_as_rectangle = ShapeFactory.shape_as_rectangle(shape=shape)
        except ValueError:
            # Thrown if the resulting bounding box is invalid
            return False

        # The min and max are to handle out-of-bounds coordinates
        x1 = min(max(round(shape_as_rectangle.x1 * media_width), 0), media_width)
        x2 = min(max(round(shape_as_rectangle.x2 * media_width), 0), media_width)
        y1 = min(max(round(shape_as_rectangle.y1 * media_height), 0), media_height)
        y2 = min(max(round(shape_as_rectangle.y2 * media_height), 0), media_height)

        if x1 == x2 or y1 == y2:
            is_valid = False
        return is_valid
