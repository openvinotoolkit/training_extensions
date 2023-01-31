"""This UnitTest tests ShapeFactory functionality"""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


import pytest

from otx.api.entities.shapes.ellipse import Ellipse
from otx.api.entities.shapes.polygon import Point, Polygon
from otx.api.entities.shapes.rectangle import Rectangle
from otx.api.utils.shape_factory import ShapeFactory
from tests.unit.api.constants.components import OtxSdkComponent
from tests.unit.api.constants.requirements import Requirements


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestShapeFactory:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_rectangle_shape_conversion(self):
        """
        <b>Description:</b>
        Checks that conversions from Rectangle to other shapes works correctly

        <b>Input data:</b>
        A rectangle at [0.25, 0.1, 0.5, 0.3]

        <b>Expected results:</b>
        The test passes if the rectangle can be converted to Ellipse and Polygon

        <b>Steps</b>
        1. Create rectangle and get coordinates
        2. Convert to Ellipse
        3. Convert to Polygon
        4. Convert to Rectangle
        """
        rectangle = Rectangle(x1=0.25, y1=0.1, x2=0.5, y2=0.3)
        rectangle_coords = (rectangle.x1, rectangle.y1, rectangle.x2, rectangle.y2)

        ellipse = ShapeFactory.shape_as_ellipse(rectangle)
        assert isinstance(ellipse, Ellipse)
        assert (ellipse.x1, ellipse.y1, ellipse.x2, ellipse.y2) == rectangle_coords

        polygon = ShapeFactory.shape_as_polygon(rectangle)
        assert isinstance(polygon, Polygon)
        assert (
            polygon.min_x,
            polygon.min_y,
            polygon.max_x,
            polygon.max_y,
        ) == rectangle_coords

        rectangle2 = ShapeFactory.shape_as_rectangle(rectangle)
        assert isinstance(rectangle2, Rectangle)
        assert rectangle == rectangle2

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_ellipse_shape_conversion(self):
        """
        <b>Description:</b>
        Checks that conversions from Ellipse to other shapes works correctly

        <b>Input data:</b>
        A rectangle at [0.1, 0.1, 0.5, 0.2]

        <b>Expected results:</b>
        The test passes if the Ellipse can be converted to Rectangle and Polygon

        <b>Steps</b>
        1. Create Ellipse and get coordinates
        2. Convert to Ellipse
        3. Convert to Polygon
        4. Convert to Rectangle
        """
        ellipse = Ellipse(x1=0.1, y1=0.1, x2=0.5, y2=0.2)
        ellipse_coords = (ellipse.x1, ellipse.y1, ellipse.x2, ellipse.y2)

        ellipse2 = ShapeFactory.shape_as_ellipse(ellipse)
        assert isinstance(ellipse, Ellipse)
        assert ellipse == ellipse2

        polygon = ShapeFactory.shape_as_polygon(ellipse)
        assert isinstance(polygon, Polygon)
        assert polygon.min_x == pytest.approx(ellipse.x1, 0.1)
        assert polygon.min_y == pytest.approx(ellipse.y1, 0.1)
        assert polygon.max_x == pytest.approx(ellipse.x2, 0.1)
        assert polygon.max_y == pytest.approx(ellipse.y2, 0.1)

        rectangle = ShapeFactory.shape_as_rectangle(ellipse)
        assert isinstance(rectangle, Rectangle)
        assert (
            rectangle.x1,
            rectangle.y1,
            rectangle.x2,
            rectangle.y2,
        ) == ellipse_coords

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_polygon_shape_conversion(self):
        """
        <b>Description:</b>
        Checks that conversions from Polygon to other shapes works correctly

        <b>Input data:</b>
        A Polygon at [[0.01, 0.2], [0.35, 0.2], [0.35, 0.4]]

        <b>Expected results:</b>
        The test passes if the Polygon can be converted to Rectangle and Ellipse

        <b>Steps</b>
        1. Create rectangle and get coordinates
        2. Convert to Ellipse
        3. Convert to Polygon
        4. Convert to Rectangle
        """
        point1 = Point(x=0.01, y=0.2)
        point2 = Point(x=0.35, y=0.2)
        point3 = Point(x=0.35, y=0.4)
        polygon = Polygon(points=[point1, point2, point3])
        polygon_coords = (polygon.min_x, polygon.min_y, polygon.max_x, polygon.max_y)

        ellipse = ShapeFactory.shape_as_ellipse(polygon)
        assert isinstance(ellipse, Ellipse)
        assert (ellipse.x1, ellipse.y1, ellipse.x2, ellipse.y2) == polygon_coords

        polygon2 = ShapeFactory.shape_as_polygon(polygon)
        assert isinstance(polygon2, Polygon)
        assert polygon == polygon2

        rectangle = ShapeFactory.shape_as_rectangle(polygon)
        assert isinstance(rectangle, Rectangle)
        assert (
            rectangle.x1,
            rectangle.y1,
            rectangle.x2,
            rectangle.y2,
        ) == polygon_coords

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_produces_valid_crop(self):
        """
        <b>Description:</b>
        Checks that shape_produces_valid_crop returns the correct values and
        does not raise errors

        <b>Input data:</b>
        A valid Rectangle at [0, 0.4, 1, 0.5]
        A Polygon that has an invalid bounding box

        <b>Expected results:</b>
        The test passes if the call with the Rectangle returns True and
        the one with the polygon returns False

        <b>Steps</b>
        1. Check Valid Rectangle
        2. Check invalid Polygon
        """
        rectangle = Rectangle(x1=0, y1=0.4, x2=1, y2=0.5)
        assert ShapeFactory.shape_produces_valid_crop(rectangle, 100, 100)

        point1 = Point(x=0.01, y=0.1)
        point2 = Point(x=0.35, y=0.1)
        point3 = Point(x=0.35, y=0.1)
        polygon = Polygon(points=[point1, point2, point3])
        assert not ShapeFactory.shape_produces_valid_crop(polygon, 100, 250)
