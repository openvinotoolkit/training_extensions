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

import pytest

from ote_sdk.entities.color import Color, ColorEntity

from ote_sdk.tests.constants.ote_sdk_components import OteSdkComponent
from ote_sdk.tests.constants.requirements import Requirements

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

@pytest.mark.components(OteSdkComponent.OTE_SDK)
class TestColor:

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_color(self):
        """
        <b>Description:</b>
        Check that Color can correctly return the value

        <b>Expected results:</b>
        Test passes if the results matches
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

@pytest.mark.components(OteSdkComponent.OTE_SDK)
class TestColorEntity:

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_color_entity(self):
        """
        <b>Description:</b>
        Check that ColorEntity rizes some methods

        <b>Expected results:</b>
        Test passes if the rise NotImplementedError
        """

        color_entity = ColorEntity

        with pytest.raises(NotImplementedError):
            color_entity.hex_str.__get__(property)

        with pytest.raises(NotImplementedError):
            color_entity.random()

        with pytest.raises(NotImplementedError):
            color_entity.from_hex_str(color_hex)
