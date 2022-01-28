"""This module tests classes related to InferenceParameters"""

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

import dataclasses

import pytest

from ote_sdk.entities.inference_parameters import InferenceParameters
from ote_sdk.tests.constants.ote_sdk_components import OteSdkComponent
from ote_sdk.tests.constants.requirements import Requirements


@pytest.mark.components(OteSdkComponent.OTE_SDK)
class TestInferenceParameters:
    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_inference_parameters_members(self):
        """
        <b>Description:</b>
        To test InferenceParameters dataclass members

        <b>Input data:</b>
        Initialized instance of InferenceParameters class

        <b>Expected results:</b>

        <b>Steps</b>
        1. Create InferenceParameters
        2. Check members
        """
        infer_params = InferenceParameters()

        assert dataclasses.is_dataclass(infer_params)
        assert len(dataclasses.fields(infer_params)) == 2
        assert dataclasses.fields(infer_params)[0].name == "is_evaluation"
        assert dataclasses.fields(infer_params)[1].name == "update_progress"
        assert type(infer_params.is_evaluation) is bool
        assert callable(infer_params.update_progress)
        with pytest.raises(AttributeError):
            str(infer_params.WRONG)

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_inference_parameters_update_member(self):
        """
        <b>Description:</b>
        To test InferenceParameters dataclass members update

        <b>Input data:</b>
        Initialized instance of InferenceParameters class

        <b>Expected results:</b>

        <b>Steps</b>
        1. Create InferenceParameters
        2. Check members update
        """
        infer_params = InferenceParameters(False)
        assert infer_params.is_evaluation is False
        assert (
            infer_params.update_progress(-2147483648) is infer_params.update_progress(0)
            and infer_params.update_progress(2147483648) is None
        )

        infer_params = InferenceParameters(True)
        assert infer_params.is_evaluation is True
        assert (
            infer_params.update_progress(-2147483648) is infer_params.update_progress(0)
            and infer_params.update_progress(2147483648) is None
        )
