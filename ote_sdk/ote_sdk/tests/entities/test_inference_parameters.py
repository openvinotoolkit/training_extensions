"""This module tests classes related to InferenceParameters"""

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
import dataclasses

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
                Initiated instance of InferenceParameters class

                <b>Expected results:</b>

                <b>Steps</b>
                1. Initiate InferenceParameters instance
                2. Check members
                """
        infer_params = InferenceParameters()

        assert dataclasses.is_dataclass(infer_params)
        assert len(dataclasses.fields(infer_params)) == 2
        assert dataclasses.fields(infer_params)[0].name == 'is_evaluation'
        assert dataclasses.fields(infer_params)[1].name == 'update_progress'
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
                Initiated instance of InferenceParameters class

                <b>Expected results:</b>

                <b>Steps</b>
                1. Initiate InferenceParameters instance
                2. Check members update
                """
        infer_params = InferenceParameters(False)
        assert infer_params.is_evaluation is False
        assert infer_params.update_progress(-2147483648) is infer_params.update_progress(0) \
               and infer_params.update_progress(2147483648) is None

        infer_params = InferenceParameters(True)
        assert infer_params.is_evaluation is True
        assert infer_params.update_progress(-2147483648) is infer_params.update_progress(0) \
               and infer_params.update_progress(2147483648) is None
