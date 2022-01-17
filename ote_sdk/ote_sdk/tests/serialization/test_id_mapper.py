#
# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

""" This module contains tests for the mapper for ID entities """

import pytest

from ote_sdk.entities.id import ID
from ote_sdk.serialization.id_mapper import IDMapper
from ote_sdk.tests.constants.ote_sdk_components import OteSdkComponent
from ote_sdk.tests.constants.requirements import Requirements


@pytest.mark.components(OteSdkComponent.OTE_SDK)
class TestIDMapper:
    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_serialized_representiaton(self):
        """
        This test serializes ID and checks serialized representation.
        """

        id = ID("21434231456")
        serialized_id = IDMapper.forward(id)
        assert serialized_id == "21434231456"

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_serialization_deserialization(self):
        """
        This test serializes ID, deserializes serialized ID and compare with original.
        """

        id = ID("21434231456")
        serialized_id = IDMapper.forward(id)
        deserialized_id = IDMapper.backward(serialized_id)
        assert id == deserialized_id
