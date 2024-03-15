#
# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

""" This module contains tests for the mapper for ID entities """

import pytest

from otx.api.entities.id import ID
from otx.api.serialization.id_mapper import IDMapper
from tests.unit.api.constants.components import OtxSdkComponent
from tests.unit.api.constants.requirements import Requirements


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestIDMapper:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_serialized_representiaton(self):
        """
        This test serializes ID and checks serialized representation.
        """

        id_ = ID("21434231456")
        serialized_id = IDMapper.forward(id_)
        assert serialized_id == "21434231456"

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_serialization_deserialization(self):
        """
        This test serializes ID, deserializes serialized ID and compare with original.
        """

        id_ = ID("21434231456")
        serialized_id = IDMapper.forward(id_)
        deserialized_id = IDMapper.backward(serialized_id)
        assert id_ == deserialized_id
