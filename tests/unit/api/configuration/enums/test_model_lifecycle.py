# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest

from otx.api.configuration.enums.model_lifecycle import ModelLifecycle
from tests.unit.api.constants.components import OtxSdkComponent
from tests.unit.api.constants.requirements import Requirements


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestModelLifecycle:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_model_lifecycle(self):
        """
        <b>Description:</b>
        Check ModelLifecycle Enum class elements

        <b>Expected results:</b>
        Test passes if ModelLifecycle Enum class length is equal to expected value and its elements have expected
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
