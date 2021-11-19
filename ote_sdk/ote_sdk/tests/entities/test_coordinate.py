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

from ote_sdk.entities.coordinate import Coordinate
from ote_sdk.tests.constants.ote_sdk_components import OteSdkComponent
from ote_sdk.tests.constants.requirements import Requirements


@pytest.mark.components(OteSdkComponent.OTE_SDK)
class TestCoordinate:
    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_coordinate(self):
        """
        <b>Description:</b>
        To test Coordinate class

        <b>Input data:</b>
        Three coordinate instances

        <b>Expected results:</b>
        1. It raises TypeError in case attempt of initiation with wrong parameters numbers
        2. Fields of instances initiated with correct values
        3. repr method returns correct strings then used against each instance
        4. hash method works as expected
        5. as_tuple method works as expected
        6. as__int_tuple method works as expected
        7. '==' method works is expected

        <b>Steps</b>
        1. Attempt to initiate class instance with wrong parameters numbers
        2. Initiate three class instances:
            two of them with similar set of init values, third one with different one.
        3. Check repr method
        4. Check hash method
        5. Check as_tuple() method
        6. Check as__int_tuple() method
        7. Check __eq__ method

        """
        with pytest.raises(TypeError):
            Coordinate()

        with pytest.raises(TypeError):
            Coordinate(1)

        coord_a = Coordinate(x=0, y=0)
        coord_b = Coordinate(x=0., y=.0)
        coord_c = Coordinate(x=1, y=1)

        assert isinstance(coord_a, Coordinate)
        assert coord_a.x == 0
        assert coord_a.y == 0
        assert repr(coord_a) == "Coordinate(x=0, y=0)"
        assert hash(coord_a) == coord_a.__hash__()
        assert coord_a.as_tuple() == (0, 0)
        assert coord_a.as_int_tuple() == (0, 0)

        assert isinstance(coord_b, Coordinate)
        assert coord_b.x == 0.0
        assert coord_b.y == 0.0
        assert repr(coord_b) == "Coordinate(x=0.0, y=0.0)"
        assert hash(coord_b) == coord_b.__hash__()
        assert coord_b.as_tuple() == (0.0, 0.0)
        assert coord_b.as_int_tuple() == (0, 0)

        assert isinstance(coord_c, Coordinate)
        assert coord_c.x == 1
        assert coord_c.y == 1
        assert repr(coord_c) == "Coordinate(x=1, y=1)"
        assert hash(coord_c) == coord_c.__hash__()
        assert coord_c.as_tuple() == (1, 1)
        assert coord_c.as_int_tuple() == (1, 1)

        assert coord_a == coord_b != coord_c

