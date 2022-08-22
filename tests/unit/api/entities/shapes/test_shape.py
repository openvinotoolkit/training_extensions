# Copyright (C) 2020-2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import itertools
import warnings
from datetime import datetime

import pytest

from otx.api.entities.shapes.ellipse import Ellipse
from otx.api.entities.shapes.polygon import Point, Polygon
from otx.api.entities.shapes.rectangle import Rectangle
from otx.api.entities.shapes.shape import (
    GeometryException,
    Shape,
    ShapeEntity,
    ShapeType,
)
from tests.unit.api.constants.components import OtxSdkComponent
from tests.unit.api.constants.requirements import Requirements


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestShapeType:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_shapetype(self):
        """
        <b>Description:</b>
        Check ShapeType class length returns expected value

        <b>Expected results:</b>
        Test passes if ShapeType enum class length equal expected value
        """
        assert len(ShapeType) == 3
        assert ShapeType.ELLIPSE.value == 1
        assert ShapeType.RECTANGLE.value == 2
        assert ShapeType.POLYGON.value == 3


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestShapeEntity:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_shape_entity_not_implemented_methods(self):
        """
        <b>Description:</b>
        Check not implemented methods of ShapeEntity class

        <b>Expected results:</b>
        Test passes if NotImplementedError exception raises when using not implemented methods on ShapeEntity instance
        """
        rectangle_entity = Rectangle(x1=0.2, y1=0.2, x2=0.6, y2=0.7)
        ellipse_entity = Ellipse(x1=0.4, y1=0.1, x2=0.9, y2=0.8)
        polygon_entity = Polygon(
            [
                Point(0.3, 0.4),
                Point(0.3, 0.7),
                Point(0.5, 0.75),
                Point(0.8, 0.7),
                Point(0.8, 0.4),
            ]
        )
        for shape in [rectangle_entity, ellipse_entity, polygon_entity]:
            with pytest.raises(NotImplementedError):
                ShapeEntity.get_area(shape)
            with pytest.raises(NotImplementedError):
                ShapeEntity.intersects(shape, shape)
            with pytest.raises(NotImplementedError):
                ShapeEntity.contains_center(shape, shape)
            with pytest.raises(NotImplementedError):
                ShapeEntity.normalize_wrt_roi_shape(shape, rectangle_entity)
            with pytest.raises(NotImplementedError):
                ShapeEntity.denormalize_wrt_roi_shape(shape, rectangle_entity)
            with pytest.raises(NotImplementedError):
                ShapeEntity._as_shapely_polygon(shape)


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestShape:
    @staticmethod
    def fully_covering_rectangle() -> Rectangle:
        return Rectangle.generate_full_box()

    @staticmethod
    def fully_covering_ellipse() -> Ellipse:
        return Ellipse(x1=0.0, y1=0.0, x2=1.0, y2=1.0)

    @staticmethod
    def fully_covering_polygon() -> Polygon:
        return Polygon(
            [
                Point(0.0, 0.1),
                Point(0.0, 0.9),
                Point(0.5, 1.0),
                Point(1.0, 0.9),
                Point(1.0, 0.0),
                Point(0.0, 0.1),
            ]
        )

    def rectangle(self) -> Rectangle:
        return Rectangle(x1=0.2, y1=0.2, x2=0.6, y2=0.7)

    def ellipse(self) -> Ellipse:
        return Ellipse(x1=0.4, y1=0.1, x2=0.9, y2=0.8)

    def polygon(self) -> Polygon:
        return Polygon(
            [
                Point(0.3, 0.4),
                Point(0.3, 0.7),
                Point(0.5, 0.75),
                Point(0.8, 0.7),
                Point(0.8, 0.4),
                Point(0.3, 0.4),
            ],
        )

    @staticmethod
    def not_inscribed_rectangle() -> Rectangle:
        return Rectangle(x1=0.0, y1=0.0, x2=0.01, y2=0.01)

    @staticmethod
    def not_inscribed_ellipse() -> Ellipse:
        return Ellipse(x1=0.0, y1=0.0, x2=0.01, y2=0.01)

    @staticmethod
    def not_inscribed_polygon() -> Polygon:
        return Polygon(
            [
                Point(0.0, 0.0),
                Point(0.0, 0.01),
                Point(0.01, 0.02),
                Point(0.02, 0.01),
                Point(0.02, 0.0),
                Point(0.0, 0.0),
            ]
        )

    @staticmethod
    def base_self_intersect_polygon() -> Polygon:
        return Polygon(
            [
                Point(0.3, 0.3),
                Point(0.4, 0.3),
                Point(0.3, 0.3),
                Point(0.3, 0.2),
                Point(0.3, 1),
                Point(0.2, 0.2),
            ]
        )

    @staticmethod
    def other_self_intersect_polygon() -> Polygon:
        return Polygon(
            [
                Point(0.3, 0.2),
                Point(0.2, 0.3),
                Point(0.3, 0.1),
                Point(0.3, 0.2),
                Point(0, 0.2),
                Point(0, 4),
            ]
        )

    @staticmethod
    def lower_side_intersect_shapes() -> list:
        return [
            Rectangle(x1=0.2, y1=0.1, x2=0.5, y2=0.4),
            Polygon(
                [
                    Point(0.35, 0.1),
                    Point(0.2, 0.2),
                    Point(0.2, 0.4),
                    Point(0.5, 0.4),
                    Point(0.5, 0.2),
                    Point(0.35, 0.1),
                ]
            ),
        ]

    @staticmethod
    def upper_side_intersect_shapes() -> list:
        return [
            Rectangle(x1=0.2, y1=0.4, x2=0.5, y2=0.7),
            Polygon(
                [
                    Point(0.35, 0.7),
                    Point(0.2, 0.6),
                    Point(0.2, 0.4),
                    Point(0.5, 0.4),
                    Point(0.5, 0.6),
                    Point(0.35, 0.7),
                ]
            ),
        ]

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_shape_get_area_method(self):
        """
        <b>Description:</b>
        Check get_area not implemented method of Shape class

        <b>Expected results:</b>
        Test passes if NotImplementedError exception raised when using get_area method on Shape instance
        """
        for shape in [self.rectangle(), self.ellipse(), self.polygon()]:
            with pytest.raises(NotImplementedError):
                Shape.get_area(shape)

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_shape_magic_methods(self):
        """
        <b>Description:</b>
        Check __repr__ and __hash__ methods for Shape class instance

        <b>Expected results:</b>
        Test passes if __repr__ and __hash__ method return expected values

        <b>Steps</b>
        1. Check that __repr__ method returns expected value
        2. Check that __hash__ method returns expected value
        """
        test_rectangle = Rectangle(0.0, 0.0, 1.0, 1.0, modification_date=datetime(year=2021, month=11, day=23))
        assert Shape.__repr__(test_rectangle) == "Shape with modification date:('2021-11-23 00:00:00')"
        assert Shape.__hash__(test_rectangle) == hash(str(test_rectangle))

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_shape_intersects(self):
        """
        <b>Description:</b>
        Check Shape intersects method for Rectangle, Ellipse and Polygon objects

        <b>Expected results:</b>
        Test passes if intersects method returns expected values

        <b>Steps</b>
        1. Check intersects method when Shapes intersect completely
        2. Check intersects method when Shapes intersect in several points
        3. Check intersects method when Shapes intersect by one side
        4. Check intersects method when Shapes not intersect
        5. Check GeometryException exception raised with incorrect parameters for intersect method
        """
        inscribed_shapes_list = [self.rectangle(), self.ellipse(), self.polygon()]
        # Check when Shapes intersect fully
        for full_element in [
            self.fully_covering_rectangle(),
            self.fully_covering_ellipse(),
            self.fully_covering_polygon(),
        ]:
            for inscribed in inscribed_shapes_list:
                assert full_element.intersects(inscribed)
                assert inscribed.intersects(full_element)
        # Check when Shapes intersect in several points
        for shape, other_shape in list(itertools.combinations(inscribed_shapes_list, 2)):
            assert shape.intersects(other_shape)
            assert other_shape.intersects(shape)
        # Check when Shapes intersect by one side
        for upper_shape in self.upper_side_intersect_shapes():
            for lower_shape in self.lower_side_intersect_shapes():
                assert lower_shape.intersects(upper_shape)
        # Check when Shapes not intersect
        for shape in inscribed_shapes_list:
            for not_inscribed_shape in (
                self.not_inscribed_rectangle(),
                self.not_inscribed_ellipse(),
                self.not_inscribed_polygon(),
            ):
                assert not shape.intersects(not_inscribed_shape)
                assert not not_inscribed_shape.intersects(shape)
        # Checking GeometryException exception raised
        with pytest.raises(GeometryException):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "Polygon coordinates")
                self.base_self_intersect_polygon().intersects(self.other_self_intersect_polygon())

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_shape_contains_center(self):
        """
        <b>Description:</b>
        Check Shape contains_center method for Rectangle, Ellipse and Polygon objects

        <b>Expected results:</b>
        Test passes if contains_center method returns expected values

        <b>Steps</b>
        1. Check contains_center method when a Polygon, Rectangle and Ellipse fall within a Rectangle
        2. Check contains_center method when a Polygon, Rectangle and Ellipse fall outside a Rectangle
        """
        rectangle_full = self.fully_covering_rectangle()
        shapes_inside = [self.polygon(), self.ellipse(), self.rectangle()]

        rectangle_part = self.rectangle()
        shapes_outside = [
            self.not_inscribed_polygon(),
            self.not_inscribed_ellipse(),
            self.not_inscribed_rectangle(),
        ]

        for shape_inside in shapes_inside:
            assert rectangle_full.contains_center(shape_inside)
        for shape_outside in shapes_outside:
            assert not rectangle_part.contains_center(shape_outside)

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_validate_coordinates(self):
        """
        <b>Description:</b>
        Check Shape validate_coordinates method for Rectangle, Ellipse and Polygon objects

        <b>Expected results:</b>
        Test passes if validate_coordinates method returns expected values

        <b>Steps</b>
        1. Check validate_coordinates method for Shapes with 0.0<=x,y<=1.0
        2. Check validate_coordinates method for Shapes with x<0.0
        3. Check validate_coordinates method for Shapes with x>1.0
        4. Check validate_coordinates method for Shapes with y<0.0
        5. Check validate_coordinates method for Shapes with y>1.0
        6. Check validate_coordinates method for Shapes with x,y<0.0
        7. Check validate_coordinates method for Shapes with x,y>1.0
        8. Check validate_coordinates method for Shapes with x>1.0, y<0.0
        9. Check validate_coordinates method for Shapes with x<1.0, y>1.0
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", r".* coordinates")
            for shape in [self.rectangle(), self.ellipse(), self.polygon()]:
                assert shape._validate_coordinates(x=0.0, y=0.0)
                assert shape._validate_coordinates(x=1.0, y=1.0)
                assert shape._validate_coordinates(x=0.2, y=0.3)
                assert not shape._validate_coordinates(x=-0.1, y=0.0)
                assert not shape._validate_coordinates(x=1.1, y=1.0)
                assert not shape._validate_coordinates(x=0.2, y=-0.3)
                assert not shape._validate_coordinates(x=0.2, y=1.3)
                assert not shape._validate_coordinates(x=-0.1, y=-0.2)
                assert not shape._validate_coordinates(x=1.1, y=1.2)
                assert not shape._validate_coordinates(x=1.2, y=-0.3)
                assert not shape._validate_coordinates(x=-1.2, y=1.3)
