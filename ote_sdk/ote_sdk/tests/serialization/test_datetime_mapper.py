#
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
