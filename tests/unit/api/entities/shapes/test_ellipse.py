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

from otx.api.entities.shapes.ellipse import Ellipse
from otx.api.entities.shapes.rectangle import Rectangle
from otx.api.utils.time_utils import now
from tests.unit.api.constants.components import OtxSdkComponent
from tests.unit.api.constants.requirements import Requirements


@pytest.mark.components(OtxSdkComponent.OTX_API)
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

    @staticmethod
    def width_gt_height_ellipse_params():
        width_gt_height_ellipse_params = {
            "x1": 0.5,
            "x2": 0.8,
            "y1": 0.1,
            "y2": 0.3,
        }
        return width_gt_height_ellipse_params

    @pytest.mark.priority_medium
    @pytest.mark.unit
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
        assert width_lt_height_ellipse.width == pytest.approx(0.09999999999999998)
        assert width_lt_height_ellipse.height == pytest.approx(0.10000000000000003)
        assert width_lt_height_ellipse.x_center == pytest.approx(0.45)
        assert width_lt_height_ellipse.y_center == pytest.approx(0.35)
        assert width_lt_height_ellipse.minor_axis == pytest.approx(0.04999999999999999)
        assert width_lt_height_ellipse.major_axis == pytest.approx(0.05000000000000002)

        width_gt_height_ellipse = Ellipse(**self.width_gt_height_ellipse_params())
        assert width_gt_height_ellipse.height < width_gt_height_ellipse.width
        assert width_gt_height_ellipse.minor_axis == width_gt_height_ellipse.height / 2
        assert width_gt_height_ellipse.width == pytest.approx(0.30000000000000004)
        assert width_gt_height_ellipse.height == pytest.approx(0.19999999999999998)
        assert width_gt_height_ellipse.x_center == pytest.approx(0.65)
        assert width_gt_height_ellipse.y_center == pytest.approx(0.2)
        assert width_gt_height_ellipse.minor_axis == pytest.approx(0.09999999999999999)
        assert width_gt_height_ellipse.major_axis == pytest.approx(0.15000000000000002)

    @pytest.mark.priority_medium
    @pytest.mark.unit
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
    @pytest.mark.unit
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
    @pytest.mark.unit
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
    @pytest.mark.unit
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
        ellipse = self.ellipse()
        number_of_coordinates = 3
        coordinates_ellipse_line = ellipse.get_evenly_distributed_ellipse_coordinates(number_of_coordinates)
        assert len(coordinates_ellipse_line) == 3
        assert coordinates_ellipse_line[0] == pytest.approx((1.0, 0.25))
        assert coordinates_ellipse_line[1] == pytest.approx((0.625, 0.4665063509461097))
        assert coordinates_ellipse_line[2] == pytest.approx((0.6249999999999999, 0.033493649053890406))

        width_gt_height_ellipse = Ellipse(**self.width_gt_height_ellipse_params())
        coordinates_ellipse_line = width_gt_height_ellipse.get_evenly_distributed_ellipse_coordinates(
            number_of_coordinates
        )
        assert width_gt_height_ellipse.height < width_gt_height_ellipse.width
        assert len(coordinates_ellipse_line) == 3
        assert coordinates_ellipse_line[0] == pytest.approx((0.65, 0.3))
        assert coordinates_ellipse_line[1] == pytest.approx((0.7666223198362645, 0.1371094972158116))
        assert coordinates_ellipse_line[2] == pytest.approx((0.5333776801637811, 0.13710949721577403))

    @pytest.mark.priority_medium
    @pytest.mark.unit
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
        assert shapely_polygon.area == pytest.approx(0.1958331774442254)

    @pytest.mark.priority_medium
    @pytest.mark.unit
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
        assert area == pytest.approx(0.19634954084936207)
