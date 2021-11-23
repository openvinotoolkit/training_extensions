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
        batch_size_default = mc.learning_parameters.get_metadata("batch_size")["default_value"]

        assert mc.learning_parameters.epochs == epoch_default
        assert mc.learning_parameters.batch_size == batch_size_default

        mc.learning_parameters.epochs = epoch_default + 5
        mc.learning_parameters.batch_size = batch_size_default + 4

        assert mc.learning_parameters.batch_size == batch_size_default + 4
        assert mc.learning_parameters.epochs == epoch_default + 5
