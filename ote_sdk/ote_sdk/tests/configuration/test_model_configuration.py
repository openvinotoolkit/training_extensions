# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest

from ote_sdk.configuration.default_model_parameters import DefaultModelParameters
from ote_sdk.tests.constants.ote_sdk_components import OteSdkComponent
from ote_sdk.tests.constants.requirements import Requirements


@pytest.mark.components(OteSdkComponent.OTE_SDK)
class TestModelConfiguration:
    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_model_configuration(self):
        mc = DefaultModelParameters()
        assert hasattr(mc, "learning_parameters")

        epoch_default = mc.learning_parameters.get_metadata("epochs")["default_value"]
        batch_size_default = mc.learning_parameters.get_metadata("batch_size")[
            "default_value"
        ]

        assert mc.learning_parameters.epochs == epoch_default
        assert mc.learning_parameters.batch_size == batch_size_default

        mc.learning_parameters.epochs = epoch_default + 5
        mc.learning_parameters.batch_size = batch_size_default + 4

        assert mc.learning_parameters.batch_size == batch_size_default + 4
        assert mc.learning_parameters.epochs == epoch_default + 5
