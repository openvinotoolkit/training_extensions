"""This module tests classes related to OptimizationParameters"""

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
                Initiated instance of OptimizationParameters class

                <b>Expected results:</b>

                <b>Steps</b>
                1. Initiate OptimizationParameters instance
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
                Initiated instance of OptimizationParameters class

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

