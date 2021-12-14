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

from ote_sdk.entities.train_parameters import (
    TrainParameters,
    default_progress_callback,
    default_save_model_callback,
)
from ote_sdk.tests.constants.ote_sdk_components import OteSdkComponent
from ote_sdk.tests.constants.requirements import Requirements


@pytest.mark.components(OteSdkComponent.OTE_SDK)
class TestTrainParameters:
    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_train_parameters(self):
        """
        <b>Description:</b>
        Check TrainParameters class object attributes

        <b>Expected results:</b>
        Test passes if attributes of TrainParameters class object are equal to expected

        <b>Steps</b>
        1. Check attributes of TrainParameters object initialized with default parameters
        2. Check attributes of TrainParameters object initialized with "resume", "update_progress" and "save_model"
        parameters
        """
        # Checking attributes of TrainParameters object initiated with default parameters
        default_values_train_parameters = TrainParameters()
        assert not default_values_train_parameters.resume
        # Expected that update_progress equal to function for default parameters TrainParameters object
        assert (
            default_values_train_parameters.update_progress == default_progress_callback
        )
        # Expected that save_model is equal to function for default parameters TrainParameters object
        assert default_values_train_parameters.save_model == default_save_model_callback
        # Checking attributes of TrainParameters object initiated with specified
        progress_callback = default_progress_callback(99.9, 99.9)
        save_model_callback = default_save_model_callback()
        specified_values_train_parameters = TrainParameters(
            resume=True,
            update_progress=progress_callback,
            save_model=save_model_callback,
        )
        assert specified_values_train_parameters.resume
        assert specified_values_train_parameters.update_progress == progress_callback
        assert specified_values_train_parameters.save_model == save_model_callback
