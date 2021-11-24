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
from datetime import datetime

import pytest

from ote_sdk.entities.color import Color
from ote_sdk.entities.label import LabelEntity
from ote_sdk.entities.scored_label import ScoredLabel, Domain
from ote_sdk.entities.shapes.ellipse import Ellipse
from ote_sdk.entities.shapes.polygon import Polygon, Point
from ote_sdk.entities.shapes.rectangle import Rectangle
from ote_sdk.entities.shapes.shape import ShapeType, GeometryException, Shape, ShapeEntity
from ote_sdk.tests.constants.ote_sdk_components import OteSdkComponent
from ote_sdk.tests.constants.requirements import Requirements


@pytest.mark.components(OteSdkComponent.OTE_SDK)
class TestShapeType:
    @pytest.mark.priority_medium
    @pytest.mark.component
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


@pytest.mark.components(OteSdkComponent.OTE_SDK)
class TestShapeEntity:
    @pytest.mark.priority_medium
    @pytest.mark.component
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
        polygon_entity = Polygon([Point(0.3, 0.4), Point(0.3, 0.7), Point(0.5, 0.75),
                                  Point(0.8, 0.7), Point(0.8, 0.4)])
        for shape in [rectangle_entity, ellipse_entity, polygon_entity]:
            with pytest.raises(NotImplementedError):
                ShapeEntity.get_area(shape)
            with pytest.raises(NotImplementedError):
                ShapeEntity.intersects(shape, shape)
            with pytest.raises(NotImplementedError):
                ShapeEntity.intersect_percentage(shape, shape)
            with pytest.raises(NotImplementedError):
                ShapeEntity.get_labels(shape)
            with pytest.raises(NotImplementedError):
                ShapeEntity.append_label(shape, ScoredLabel(LabelEntity(
                    name="classification", domain=Domain.CLASSIFICATION)))
            with pytest.raises(NotImplementedError):
                ShapeEntity.set_labels(shape, [ScoredLabel(LabelEntity(
                    name="detection", domain=Domain.DETECTION))])
            with pytest.raises(NotImplementedError):
                ShapeEntity.normalize_wrt_roi_shape(shape, rectangle_entity)
            with pytest.raises(NotImplementedError):
                ShapeEntity.denormalize_wrt_roi_shape(shape, rectangle_entity)
            with pytest.raises(NotImplementedError):
                ShapeEntity._as_shapely_polygon(shape)


@pytest.mark.components(OteSdkComponent.OTE_SDK)
class TestShape:

    @staticmethod
    def fully_covering_rectangle():
        return Rectangle.generate_full_box()

    @staticmethod
    def fully_covering_ellipse():
        return Ellipse(x1=0.0, y1=0.0, x2=1.0, y2=1.0)

    @staticmethod
    def fully_covering_polygon():
        return Polygon([Point(0.0, 0.1), Point(0.0, 0.9), Point(0.5, 1.0),
                        Point(1.0, 0.9), Point(1.0, 0.0), Point(0.0, 0.1)])

    @staticmethod
    def generate_labels_list(include_empty: bool = True):
        classification_label = ScoredLabel(LabelEntity(name="classification", domain=Domain.CLASSIFICATION,
                                                       color=Color(red=187, green=28, blue=28),
                                                       creation_date=datetime(year=2021, month=10, day=25)))
        detection_label = ScoredLabel(LabelEntity(name="detection", domain=Domain.DETECTION,
                                                  color=Color(red=180, green=30, blue=24),
                                                  creation_date=datetime(year=2021, month=9, day=24)))
        empty_label = ScoredLabel(LabelEntity(name="empty_rectangle_label", domain=Domain.CLASSIFICATION,
                                              color=Color(red=178, green=25, blue=30),
                                              creation_date=datetime(year=2021, month=7, day=26),
                                              is_empty=True))
        labels_list = [classification_label, detection_label]
        if include_empty:
            labels_list.append(empty_label)
        return labels_list

    @staticmethod
    def appendable_label(empty=False):
        return ScoredLabel(LabelEntity(name="appended_label", domain=Domain.CLASSIFICATION,
                                       color=Color(red=181, green=28, blue=31),
                                       creation_date=datetime(year=2021, month=11, day=22),
                                       is_empty=empty))

    def rectangle(self):
        return Rectangle(x1=0.2, y1=0.2, x2=0.6, y2=0.7, labels=self.generate_labels_list())

    def ellipse(self):
        return Ellipse(x1=0.4, y1=0.1, x2=0.9, y2=0.8, labels=self.generate_labels_list())

    def polygon(self):
        return Polygon([Point(0.3, 0.4), Point(0.3, 0.7), Point(0.5, 0.75),
                        Point(0.8, 0.7), Point(0.8, 0.4), Point(0.3, 0.4)], labels=self.generate_labels_list())

    @staticmethod
    def not_inscribed_rectangle():
        return Rectangle(x1=0.0, y1=0.0, x2=0.01, y2=0.01)

    @staticmethod
    def not_inscribed_ellipse():
        return Ellipse(x1=0.0, y1=0.0, x2=0.01, y2=0.01)

    @staticmethod
    def not_inscribed_polygon():
        return Polygon([Point(0.0, 0.0), Point(0.0, 0.01), Point(0.01, 0.02),
                        Point(0.02, 0.01), Point(0.02, 0.0), Point(0.0, 0.0)])

    @staticmethod
    def base_self_intersect_polygon():
        return Polygon([Point(0.3, 0.3), Point(0.4, 0.3), Point(0.3, 0.3),
                        Point(0.3, 0.2), Point(0.3, 1), Point(0.2, 0.2)])

    @staticmethod
    def other_self_intersect_polygon():
        return Polygon([Point(0.3, 0.2), Point(0.2, 0.3), Point(0.3, 0.1),
                        Point(0.3, 0.2), Point(0, 0.2), Point(0, 4)])

    @staticmethod
    def lower_side_intersect_shapes():
        return [Rectangle(x1=0.2, y1=0.1, x2=0.5, y2=0.4),
                Polygon([Point(0.35, 0.1), Point(0.2, 0.2), Point(0.2, 0.4),
                         Point(0.5, 0.4), Point(0.5, 0.2), Point(0.35, 0.1)])]

    @staticmethod
    def upper_side_intersect_shapes():
        return [Rectangle(x1=0.2, y1=0.4, x2=0.5, y2=0.7),
                Polygon([Point(0.35, 0.7), Point(0.2, 0.6), Point(0.2, 0.4),
                         Point(0.5, 0.4), Point(0.5, 0.6), Point(0.35, 0.7)])]

    @pytest.mark.priority_medium
    @pytest.mark.component
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
    @pytest.mark.component
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
    @pytest.mark.component
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
        for full_element in [self.fully_covering_rectangle(), self.fully_covering_ellipse(),
                             self.fully_covering_polygon()]:
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
                    self.not_inscribed_rectangle(), self.not_inscribed_ellipse(), self.not_inscribed_polygon()):
                assert not shape.intersects(not_inscribed_shape)
                assert not not_inscribed_shape.intersects(shape)
        # Checking GeometryException exception raised
        with pytest.raises(GeometryException):
            self.base_self_intersect_polygon().intersects(self.other_self_intersect_polygon())

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_shape_intersect_percentage(self):
        """
        <b>Description:</b>
        Check Shape intersect_percentage method for Rectangle, Ellipse and Polygon objects

        <b>Expected results:</b>
        Test passes if intersect_percentage method returns expected values

        <b>Steps</b>
        1. Check intersect_percentage method when Shapes intersect completely
        2. Check intersect_percentage method when Shapes intersect partially
        3. Check intersect_percentage method when Shapes intersect by one side
        4. Check intersect_percentage method when Shapes not intersect
        5. Check GeometryException exception raised with incorrect parameters for intersect_percentage method
        """
        inscribed_shapes_list = [self.rectangle(), self.ellipse(), self.polygon()]
        # Check when Shapes intersect completely
        for full_element in [self.fully_covering_rectangle(), self.fully_covering_ellipse(),
                             self.fully_covering_polygon()]:
            for inscribed in inscribed_shapes_list:
                assert round(full_element.intersect_percentage(inscribed), 2) == 1.0
        # Check when Shapes intersect partially
        second_rectangle = Rectangle(x1=0.3, y1=0.4, x2=0.7, y2=0.6)
        assert self.rectangle().intersect_percentage(second_rectangle) == 0.75
        assert round(self.ellipse().intersect_percentage(self.rectangle()), 2) == 0.44
        assert self.polygon().intersect_percentage(self.rectangle()) == 0.45
        # Check when Shapes intersect by one side
        for upper_shape in self.upper_side_intersect_shapes():
            for lower_shape in self.lower_side_intersect_shapes():
                assert lower_shape.intersect_percentage(upper_shape) == 0.0
        # Check shen Shapes not intersect
        for shape in inscribed_shapes_list:
            for not_inscribed_shape in (
                    self.not_inscribed_rectangle(), self.not_inscribed_ellipse(), self.not_inscribed_polygon()):
                assert shape.intersect_percentage(not_inscribed_shape) == 0.0
        # Checking GeometryException exception raised
        with pytest.raises(GeometryException):
            self.base_self_intersect_polygon().intersect_percentage(self.other_self_intersect_polygon())

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_shape_get_labels(self):
        """
        <b>Description:</b>
        Check Shape get_labels method for Rectangle, Ellipse and Polygon objects

        <b>Expected results:</b>
        Test passes if get_labels method returns expected values

        <b>Steps</b>
        1. Check get_labels method for Shapes with no labels specified
        2. Check get_labels method for Shapes with specified labels and include_empty parameter set to False
        3. Check get_labels method for Shapes with specified labels and include_empty parameter set to True
        """
        # Checks for no labels specified
        for no_labels_shape in [self.fully_covering_rectangle(), self.fully_covering_ellipse(),
                                self.fully_covering_polygon()]:
            assert no_labels_shape.get_labels() == []
        # Checks for labels specified and include_empty set to False
        expected_false_include_empty_labels = self.generate_labels_list(include_empty=False)
        for false_include_empty_labels_shape in [self.rectangle(), self.ellipse(),
                                                 self.polygon()]:
            assert false_include_empty_labels_shape.get_labels() == expected_false_include_empty_labels
        # Checks for labels specified and include_empty set to True
        expected_include_empty_labels = self.generate_labels_list(include_empty=True)
        for include_empty_labels_shape in [self.rectangle(), self.ellipse(),
                                           self.polygon()]:
            assert include_empty_labels_shape.get_labels(include_empty=True) == expected_include_empty_labels

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_shape_append_label(self):
        """
        <b>Description:</b>
        Check Shape get_labels method for Rectangle, Ellipse and Polygon objects

        <b>Expected results:</b>
        Test passes if append_label method returns expected values

        <b>Steps</b>
        1. Check append_label method to add label to Shape object with no labels specified
        2. Check append_label method to add empty label to Shape object with no labels specified
        3. Check append_label method to add label to Shape object with specified labels
        4. Check append_label method to add empty label to Shape object with specified labels
        """
        appendable_label = self.appendable_label()
        empty_appendable_label = self.appendable_label(empty=True)
        # Check for adding label to Shape with no labels specified
        for no_labels_shape in [self.fully_covering_rectangle(), self.fully_covering_ellipse(),
                                self.fully_covering_polygon()]:
            no_labels_shape.append_label(appendable_label)
            assert no_labels_shape.get_labels() == [appendable_label]
        # Check for adding empty label to Shape with no labels specified
        for no_labels_shape in [self.fully_covering_rectangle(), self.fully_covering_ellipse(),
                                self.fully_covering_polygon()]:
            no_labels_shape.append_label(empty_appendable_label)
            assert no_labels_shape.get_labels() == []
            assert no_labels_shape.get_labels(include_empty=True) == [empty_appendable_label]
        # Check for adding label to Shape with labels specified
        for shape in [self.rectangle(), self.ellipse(), self.polygon()]:
            expected_labels = shape.get_labels()
            expected_labels.append(appendable_label)
            shape.append_label(appendable_label)
        # Check for adding empty label to Shape with labels specified
        expected_labels_false_empty = self.generate_labels_list(include_empty=False)
        expected_include_empty_labels = self.generate_labels_list(include_empty=True)
        expected_include_empty_labels.append(empty_appendable_label)
        for shape in [self.rectangle(), self.ellipse(), self.polygon()]:
            shape.append_label(empty_appendable_label)
            assert shape.get_labels() == expected_labels_false_empty
            assert shape.get_labels(include_empty=True) == expected_include_empty_labels

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_shape_set_labels(self):
        """
        <b>Description:</b>
        Check Shape set_labels method for Rectangle, Ellipse and Polygon objects

        <b>Expected results:</b>
        Test passes if set_labels method returns expected values

        <b>Steps</b>
        1. Check set_labels method to add labels list to Shape object with no labels specified
        2. Check set_labels method to add empty labels list to Shape object with no labels specified
        3. Check set_labels method to add labels list to Shape object with labels specified
        4. Check set_labels method to add empty labels list to Shape object with labels specified
        """
        not_empty_label = self.appendable_label()
        new_labels_list = [not_empty_label,
                           ScoredLabel(LabelEntity(name="new_label", domain=Domain.CLASSIFICATION,
                                                   color=Color(red=183, green=31, blue=28),
                                                   creation_date=datetime(year=2021, month=9, day=25), is_empty=True))]
        expected_not_empty_labels_list = [not_empty_label]
        # Check for adding labels list to Shape with no labels specified
        for no_labels_shape in [self.fully_covering_rectangle(), self.fully_covering_ellipse(),
                                self.fully_covering_polygon()]:
            no_labels_shape.set_labels(new_labels_list)
            assert no_labels_shape.get_labels() == expected_not_empty_labels_list
            assert no_labels_shape.get_labels(include_empty=True) == new_labels_list
        # Check for adding empty labels list to Shape with no labels specified
        for no_labels_shape in [self.fully_covering_rectangle(), self.fully_covering_ellipse(),
                                self.fully_covering_polygon()]:
            no_labels_shape.set_labels([])
            assert no_labels_shape.get_labels() == []
            assert no_labels_shape.get_labels(include_empty=True) == []
        # Check for adding labels list to Shape with labels specified
        for shape in [self.rectangle(), self.ellipse(), self.polygon()]:
            shape.set_labels(new_labels_list)
            assert shape.get_labels() == expected_not_empty_labels_list
            assert shape.get_labels(include_empty=True) == new_labels_list
        # Check for adding empty labels list to Shape with labels specified
        for shape in [self.rectangle(), self.ellipse(), self.polygon()]:
            shape.set_labels([])
            assert shape.get_labels() == []
            assert shape.get_labels(include_empty=True) == []

    @pytest.mark.priority_medium
    @pytest.mark.component
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
