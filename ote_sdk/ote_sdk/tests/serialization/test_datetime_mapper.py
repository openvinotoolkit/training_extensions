#
# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest

from ote_sdk.serialization.datetime_mapper import DatetimeMapper
from ote_sdk.tests.constants.ote_sdk_components import OteSdkComponent
from ote_sdk.tests.constants.requirements import Requirements
from ote_sdk.utils.time_utils import now


@pytest.mark.components(OteSdkComponent.OTE_SDK)
class TestDatetimeMapper:
    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_serialization_deserialization(self):
        """
        This test serializes datetime, deserializes serialized datetime and compares with original one.
        """

        original_time = now()
        serialized_time = DatetimeMapper.forward(original_time)
        assert serialized_time == original_time.strftime("%Y-%m-%dT%H:%M:%S.%f")

        deserialized_time = DatetimeMapper.backward(serialized_time)
        assert original_time == deserialized_time
