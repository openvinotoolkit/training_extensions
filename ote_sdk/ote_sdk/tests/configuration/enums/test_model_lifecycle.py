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

from ote_sdk.configuration.enums.model_lifecycle import ModelLifecycle
from ote_sdk.tests.constants.ote_sdk_components import OteSdkComponent
from ote_sdk.tests.constants.requirements import Requirements


@pytest.mark.components(OteSdkComponent.OTE_SDK)
class TestModelLifecycle:
    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_model_lifecycle(self):
        """
        <b>Description:</b>
        Check ModelLifecycle Enum class elements

        <b>Expected results:</b>
        Test passes if ModelLifecycle Enum class length equal expected value and its elements have expected
        sequence numbers and values returned by __str__ method
        """
        assert len(ModelLifecycle) == 5
        assert ModelLifecycle.NONE.value == 1
        assert str(ModelLifecycle.NONE) == "NONE"
        assert ModelLifecycle.ARCHITECTURE.value == 2
        assert str(ModelLifecycle.ARCHITECTURE) == "ARCHITECTURE"
        assert ModelLifecycle.TRAINING.value == 3
        assert str(ModelLifecycle.TRAINING) == "TRAINING"
        assert ModelLifecycle.INFERENCE.value == 4
        assert str(ModelLifecycle.INFERENCE) == "INFERENCE"
        assert ModelLifecycle.TESTING.value == 5
        assert str(ModelLifecycle.TESTING) == "TESTING"
