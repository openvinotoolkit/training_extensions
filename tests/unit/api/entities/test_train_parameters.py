# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest

from otx.api.entities.train_parameters import (
    TrainParameters,
    default_progress_callback,
    default_save_model_callback,
)
from tests.unit.api.constants.components import OtxSdkComponent
from tests.unit.api.constants.requirements import Requirements


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestTrainParameters:
    @pytest.mark.priority_medium
    @pytest.mark.unit
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
        assert default_values_train_parameters.update_progress == default_progress_callback
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
