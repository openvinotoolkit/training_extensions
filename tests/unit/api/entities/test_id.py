# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import pytest
from bson import ObjectId

from otx.api.entities.id import ID
from tests.unit.api.constants.components import OtxSdkComponent
from tests.unit.api.constants.requirements import Requirements


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestID:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_id(self):
        """
        <b>Description:</b>
        Check ID class object initialization

        <b>Input data:</b>
        ID object with specified representation parameter

        <b>Expected results:</b>
        Test passes if ID object representation property and __repr__ method return expected values

        <b>Steps</b>
        1. Check representation property and __repr__ method for ID object with not specified representation parameter
        2. Check representation property and __repr__ method for ID object with ObjectId class representation parameter
        3. Check representation property and __repr__ method for ID object with str type representation parameter
        """
        # Scenario for ID object with not specified representation parameter
        no_representation_id = ID()
        assert no_representation_id.representation == ""
        assert repr(no_representation_id.representation) == "ID()"
        # Scenario for ID object with ObjectId class representation parameter
        expected_oid = "61a8b869fb7665916a39eb95"
        oid_representation = ObjectId(expected_oid)
        oid_representation_id = ID(oid_representation)
        assert oid_representation_id.representation == "61a8b869fb7665916a39eb95"
        assert repr(oid_representation_id.representation) == "ID(61a8b869fb7665916a39eb95)"
        # Scenario for ID object with str-type representation parameter
        str_representation = " String-type representation ID_1 "
        str_representation_id = ID(str_representation)
        # Leading and trailing whitespaces should be removed, only uppercase letters should be replaced by lowercase
        assert str_representation_id.representation == "string-type representation id_1"
        assert repr(str_representation_id) == "ID(string-type representation id_1)"
