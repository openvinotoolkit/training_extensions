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

from otx.api.entities.color import Color, ColorEntity
from tests.unit.api.constants.components import OtxSdkComponent
from tests.unit.api.constants.requirements import Requirements

red = 40
red_hex = "28"
green = 210
green_hex = "d2"
blue = 43
blue_hex = "2b"
alpha = 255
alpha_hex = "ff"
color_hex = f"{red_hex}{green_hex}{blue_hex}"

color = Color.from_hex_str(color_hex)


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestColor:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_color(self):
        """
        <b>Description:</b>
        Check that Color can correctly return the value

        <b>Expected results:</b>
        Test passes if the results match
        """

        assert color == Color(red=red, green=green, blue=blue, alpha=alpha)
        assert color.hex_str == f"#{color_hex}{alpha_hex}"
        assert type(color.random()) == Color
        assert color.rgb_tuple == (red, green, blue)
        assert color.bgr_tuple == (blue, green, red)
        assert color != ColorEntity
        assert repr(color) == f"Color(red={red}, green={green}, blue={blue}, alpha={alpha})"
        assert color.red == red
        assert color.green == green
        assert color.blue == blue

        color.red = 68
        color.green = 54
        color.blue = 32
        color.alpha = 0
        assert color.hex_str == "#44362000"


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestColorEntity:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_color_entity(self):
        """
        <b>Description:</b>
        Check that ColorEntity raises some exceptions

        <b>Expected results:</b>
        Test passes if the NotImplementedError is raised
        """

        color_entity = ColorEntity

        with pytest.raises(NotImplementedError):
            color_entity.hex_str.__get__(property)

        with pytest.raises(NotImplementedError):
            color_entity.random()

        with pytest.raises(NotImplementedError):
            color_entity.from_hex_str(color_hex)
