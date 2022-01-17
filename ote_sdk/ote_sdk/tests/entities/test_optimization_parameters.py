"""This module tests classes related to OptimizationParameters"""

# Copyright (C) 2020-2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import pytest
import dataclasses

from ote_sdk.entities.optimization_parameters import OptimizationParameters
from ote_sdk.tests.constants.ote_sdk_components import OteSdkComponent
from ote_sdk.tests.constants.requirements import Requirements


@pytest.mark.components(OteSdkComponent.OTE_SDK)
class TestOptimizationParameters:
    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_optimization_parameters_members(self):
        """
        <b>Description:</b>
        To test OptimizationParameters dataclass members

        <b>Input data:</b>
        Initialized instance of OptimizationParameters class

        <b>Expected results:</b>

        <b>Steps</b>
        1. Create OptimizationParameters
        2. Check members
        """
        opt_params = OptimizationParameters()

        assert dataclasses.is_dataclass(opt_params)
        assert len(dataclasses.fields(opt_params)) == 3
        assert dataclasses.fields(opt_params)[0].name == 'resume'
        assert dataclasses.fields(opt_params)[1].name == 'update_progress'
        assert dataclasses.fields(opt_params)[2].name == 'save_model'
        assert type(opt_params.resume) is bool
        assert callable(opt_params.update_progress)
        assert callable(opt_params.save_model)
        with pytest.raises(AttributeError):
            str(opt_params.WRONG)

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_optimization_parameters_update_member(self):
        """
        <b>Description:</b>
        To test OptimizationParameters dataclass members update

        <b>Input data:</b>
        Initialized instance of OptimizationParameters class

        <b>Expected results:</b>

        <b>Steps</b>
        1. Initiate OptimizationParameters instance
        2. Check members update
        """
        opt_params = OptimizationParameters(False)
        assert opt_params.resume is False
        assert opt_params.update_progress(-2147483648) is opt_params.update_progress(0) \
               is opt_params.update_progress(2147483648) is None
        assert opt_params.save_model() is None

        opt_params = OptimizationParameters(True)
        assert opt_params.resume is True
        assert opt_params.update_progress(-2147483648) is opt_params.update_progress(0) \
               is opt_params.update_progress(2147483648) is None
        assert opt_params.save_model() is None
