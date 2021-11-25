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
"""This module tests classes related to TrainParameters"""

import pytest
import dataclasses


from ote_sdk.entities.train_parameters import TrainParameters
from ote_sdk.tests.constants.ote_sdk_components import OteSdkComponent
from ote_sdk.tests.constants.requirements import Requirements

@pytest.mark.components(OteSdkComponent.OTE_SDK)
class TestTrainParameters:
    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_train_parameters_members(self):
        """
                <b>Description:</b>
                To test TrainParameters dataclass members

                <b>Input data:</b>
                Initiated instance of TrainParameters class

                <b>Expected results:</b>

                <b>Steps</b>
                1. Initiate TrainParameters instance
                2. Check members
                """
        train_params = TrainParameters()
        assert dataclasses.is_dataclass(train_params)
        assert len(dataclasses.fields(train_params)) == 3
        assert dataclasses.fields(train_params)[0].name == 'resume'
        assert dataclasses.fields(train_params)[1].name == 'update_progress'
        assert dataclasses.fields(train_params)[2].name == 'save_model'
        assert type(train_params.resume) is bool
        assert callable(train_params.update_progress)
        assert callable(train_params.save_model)
        with pytest.raises(AttributeError):
            str(train_params.WRONG)

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_train_parameters_update_member(self):
        """
                <b>Description:</b>
                To test TrainParameters dataclass members update

                <b>Input data:</b>
                Initiated instance of TrainParameters class

                <b>Expected results:</b>

                <b>Steps</b>
                1. Initiate TrainParameters instance
                2. Check members update
                """
        train_params = TrainParameters(False)
        assert train_params.resume is False
        assert train_params.update_progress(-107754) is train_params.update_progress(0) \
               and train_params.update_progress(107754) is None
        assert train_params.save_model() is None
        
        train_params = TrainParameters(True)
        assert train_params.resume is True
        assert train_params.update_progress(-2147483648) is train_params.update_progress(0) \
               and train_params.update_progress(2147483648) is None
        assert train_params.save_model() is None