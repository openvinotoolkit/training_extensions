# Copyright (C) 2021 Intel Corporation
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

import pytest
from shapely.geometry.polygon import Polygon

from ote_sdk.entities.shapes.ellipse import Ellipse
from ote_sdk.entities.shapes.rectangle import Rectangle
from ote_sdk.tests.constants.ote_sdk_components import OteSdkComponent
from ote_sdk.tests.constants.requirements import Requirements
from ote_sdk.utils.time_utils import now


@pytest.mark.components(OteSdkComponent.OTE_SDK)
class TestEllipse:
    modification_date = now()

    def ellipse_params(self):
        ellipse_params = {
            "x1": 0.5,
            "x2": 1.0,
            "y1": 0.0,
            "y2": 0.5,
            "modification_date": self.modification_date,
        }
        return ellipse_params

    def ellipse(self):
        return Ellipse(**self.ellipse_params())

    def width_gt_height_ellipse_params(self):
        width_gt_height_ellipse_params = {
            "x1": 0.5,
            "x2": 0.8,
            "y1": 0.1,
            "y2": 0.3,
        }
        return width_gt_height_ellipse_params

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_ellipse(self):
        """
        <b>Description:</b>
        Check ellipse parameters

        <b>Input data:</b>
        Coordinates

        <b>Expected results:</b>
        Test passes if Ellipse correctly calculates parameters and returns default values

        <b>Steps</b>
        1. Check ellipse params
        2. Check ellipse default values
        3. Check ellipse with incorrect coordinates
        """

        ellipse = self.ellipse()
        modification_date = self.modification_date

        assert ellipse.width == 0.5
        assert ellipse.height == 0.5
        assert ellipse.x_center == 0.75
        assert ellipse.y_center == 0.25
        assert ellipse.minor_axis == 0.25
        assert ellipse.major_axis == 0.25
        assert ellipse._labels == []
        assert ellipse.modification_date == modification_date

        incorrect_ellipse_params = {
            "x1": 0,
            "x2": 0,
            "y1": 0,
            "y2": 0,
        }

        with pytest.raises(ValueError):
            Ellipse(**incorrect_ellipse_params)

        width_lt_height_ellipse_params = {
            "x1": 0.4,
            "x2": 0.5,
            "y1": 0.3,
            "y2": 0.4,
        }

        width_lt_height_ellipse = Ellipse(**width_lt_height_ellipse_params)
        assert width_lt_height_ellipse.height > width_lt_height_ellipse.width
        assert width_lt_height_ellipse.major_axis == width_lt_height_ellipse.height / 2
        assert round(width_lt_height_ellipse.width, 16) == 0.1
        assert round(width_lt_height_ellipse.height, 16) == 0.1
        assert round(width_lt_height_ellipse.x_center, 16) == 0.45
        assert round(width_lt_height_ellipse.y_center, 16) == 0.35
        assert round(width_lt_height_ellipse.minor_axis, 16) == 0.05
        assert round(width_lt_height_ellipse.major_axis, 16) == 0.05

        width_gt_height_ellipse = Ellipse(**self.width_gt_height_ellipse_params())
        assert width_gt_height_ellipse.height < width_gt_height_ellipse.width
        assert width_gt_height_ellipse.minor_axis == width_gt_height_ellipse.height / 2
        assert round(width_gt_height_ellipse.width, 16) == 0.3
        assert round(width_gt_height_ellipse.height, 16) == 0.2
        assert round(width_gt_height_ellipse.x_center, 16) == 0.65
        assert round(width_gt_height_ellipse.y_center, 16) == 0.2
        assert round(width_gt_height_ellipse.minor_axis, 16) == 0.1
        assert round(width_gt_height_ellipse.major_axis, 16) == 0.15

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_ellipse_magic_methods(self):
        """
        <b>Description:</b>
        Check Ellipse __repr__, __eq__, __hash__ methods

        <b>Input data:</b>
        Initialized instance of Ellipse

        <b>Expected results:</b>
        Test passes if Ellipse magic methods returns correct values

        <b>Steps</b>
        1. Initialize Ellipse instance
        2. Check returning value of magic methods
        """

        x1 = self.ellipse_params()["x1"]
        x2 = self.ellipse_params()["x2"]
        y1 = self.ellipse_params()["y1"]
        y2 = self.ellipse_params()["y2"]

        ellipse = self.ellipse()
        assert repr(ellipse) == f"Ellipse(x1={x1}, y1={y1}, x2={x2}, y2={y2})"

        other_ellipse_params = {
            "x1": 0.5,
            "x2": 1.0,
            "y1": 0.0,
            "y2": 0.5,
            "modification_date": self.modification_date,
        }

        third_ellipse_params = {
            "x1": 0.3,
            "y1": 0.5,
            "x2": 0.4,
            "y2": 0.6,
            "modification_date": self.modification_date,
        }

        other_ellipse = Ellipse(**other_ellipse_params)
        third_ellipse = Ellipse(**third_ellipse_params)

        assert ellipse == other_ellipse
        assert ellipse != third_ellipse
        assert ellipse != str

        assert hash(ellipse) == hash(str(ellipse))

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_ellipse_normalize_wrt_roi_shape(self):
        """
        <b>Description:</b>
        Check Ellipse normalize_wrt_roi_shape methods

        <b>Input data:</b>
        Initialized instance of Ellipse
        Initialized instance of Rectangle

        <b>Expected results:</b>
        Test passes if Ellipse normalize_wrt_roi_shape returns correct values

        <b>Steps</b>
        1. Initialize Ellipse instance
        2. Check returning value
        """

        ellipse = self.ellipse()
        roi = Rectangle(x1=0.0, x2=0.5, y1=0.0, y2=0.5)
        normalized = ellipse.normalize_wrt_roi_shape(roi)
        assert normalized.x1 == 0.25
        assert normalized.y1 == 0.0
        assert normalized.x2 == 0.5
        assert normalized.y2 == 0.25

        with pytest.raises(ValueError):
            ellipse.normalize_wrt_roi_shape("123")

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_ellipse_denormalize_wrt_roi_shape(self):
        """
        <b>Description:</b>
        Check Ellipse denormalize_wrt_roi_shape methods

        <b>Input data:</b>
        Initialized instance of Ellipse
        Initialized instance of Rectangle

        <b>Expected results:</b>
        Test passes if Ellipse denormalize_wrt_roi_shape returns correct values

        <b>Steps</b>
        1. Initialize Ellipse instance
        2. Check returning value
        """

        ellipse = self.ellipse()
        roi = Rectangle(x1=0.5, x2=1.0, y1=0.0, y2=1.0)
        denormalized = ellipse.denormalize_wrt_roi_shape(roi)
        assert denormalized.x1 == 0.0
        assert denormalized.y1 == 0.0
        assert denormalized.x2 == 1.0
        assert denormalized.y2 == 0.5

        with pytest.raises(ValueError):
            ellipse.denormalize_wrt_roi_shape("123")

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_ellipse_get_evenly_distributed_ellipse_coordinates(self):
        """
        <b>Description:</b>
        Check Ellipse get_evenly_distributed_ellipse_coordinates methods

        <b>Input data:</b>
        Initialized instance of Ellipse

        <b>Expected results:</b>
        Test passes if Ellipse get_evenly_distributed_ellipse_coordinates returns correct values

        <b>Steps</b>
        1. Initialize Ellipse instance
        2. Check returning value
        """

        def round_tuple_values(raw_elements: tuple):
            """Function to round tuple values to 16 digits"""
            rounded_elements = []
            for raw_element in raw_elements:
                rounded_elements.append(round(raw_element, 16))
            return tuple(rounded_elements)

        ellipse = self.ellipse()
        number_of_coordinates = 3
        coordinates_ellipse_line = ellipse.get_evenly_distributed_ellipse_coordinates(
            number_of_coordinates
        )
        assert len(coordinates_ellipse_line) == 3
        assert round_tuple_values(coordinates_ellipse_line[0]) == (1.0, 0.25)
        assert round_tuple_values(coordinates_ellipse_line[1]) == (
            0.625,
            0.4665063509461097,
        )
        assert round_tuple_values(coordinates_ellipse_line[2]) == (
            0.6249999999999999,
            0.0334936490538904,
        )

        width_gt_height_ellipse = Ellipse(**self.width_gt_height_ellipse_params())
        coordinates_ellipse_line = (
            width_gt_height_ellipse.get_evenly_distributed_ellipse_coordinates(
                number_of_coordinates
            )
        )
        assert width_gt_height_ellipse.height < width_gt_height_ellipse.width
        assert len(coordinates_ellipse_line) == 3
        assert coordinates_ellipse_line[0] == (0.65, 0.3)
        assert round_tuple_values(coordinates_ellipse_line[0]) == (0.65, 0.3)
        assert round_tuple_values(coordinates_ellipse_line[1]) == (
            0.7666223198362645,
            0.1371094972158116,
        )
        assert round_tuple_values(coordinates_ellipse_line[2]) == (
            0.5333776801637811,
            0.137109497215774,
        )

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_ellipse__as_shapely_polygon(self):
        """
        <b>Description:</b>
        Check Ellipse _as_shapely_polygon methods

        <b>Input data:</b>
        Initialized instance of Ellipse

        <b>Expected results:</b>
        Test passes if Ellipse _as_shapely_polygon returns correct values

        <b>Steps</b>
        1. Initialize Ellipse instance
        2. Check returning value
        """

        ellipse = self.ellipse()
        shapely_polygon = ellipse._as_shapely_polygon()
        assert shapely_polygon.__class__ == Polygon
        assert round(shapely_polygon.area, 16) == 0.1958331774442254

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_ellipse_get_area(self):
        """
        <b>Description:</b>
        Check Ellipse get_area methods

        <b>Input data:</b>
        Initialized instance of Ellipse

        <b>Expected results:</b>
        Test passes if Ellipse get_area returns correct values

        <b>Steps</b>
        1. Initialize Ellipse instance
        2. Check returning value
        """

        ellipse = self.ellipse()
        area = ellipse.get_area()
        assert round(area, 16) == 0.1963495408493621
