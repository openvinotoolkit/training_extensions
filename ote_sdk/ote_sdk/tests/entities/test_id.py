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
from bson import ObjectId

from ote_sdk.entities.id import ID
from ote_sdk.tests.constants.ote_sdk_components import OteSdkComponent
from ote_sdk.tests.constants.requirements import Requirements


@pytest.mark.components(OteSdkComponent.OTE_SDK)
class TestID:
    @pytest.mark.priority_medium
    @pytest.mark.component
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
        assert (
            repr(oid_representation_id.representation) == "ID(61a8b869fb7665916a39eb95)"
        )
        # Scenario for ID object with str-type representation parameter
        str_representation = " String-type representation ID_1 "
        str_representation_id = ID(str_representation)
        # Leading and trailing whitespaces should be removed, only uppercase letters should be replaced by lowercase
        assert str_representation_id.representation == "string-type representation id_1"
        assert repr(str_representation_id) == "ID(string-type representation id_1)"
