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


import pytest

from otx.api.entities.coordinate import Coordinate
from tests.unit.api.constants.components import OtxSdkComponent
from tests.unit.api.constants.requirements import Requirements


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestCoordinate:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_coordinate(self):
        """
        <b>Description:</b>
        To test Coordinate class

        <b>Input data:</b>
        Three coordinate instances

        <b>Expected results:</b>
        1. It raises TypeError in case of attempt of instantiation with wrong number of parameters
        2. Fields of instances initialized with correct values
        3. repr method returns correct strings then used against each instance
        4. hash method works as expected
        5. as_tuple method works as expected
        6. as_int_tuple method works as expected
        7. '==' method works as expected

        <b>Steps</b>
        1. Attempt to create Coordinate with wrong parameters numbers
        2. Create three Coordinate:
            two of them with similar set of init values, third one with different one.
        3. Check repr method
        4. Check hash method
        5. Check as_tuple() method
        6. Check as_int_tuple() method
        7. Check __eq__ method
        """
        with pytest.raises(TypeError):
            Coordinate()

        with pytest.raises(TypeError):
            Coordinate(1)

        coord_a = Coordinate(x=0, y=0)
        coord_b = Coordinate(x=0.0, y=0.0)
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
