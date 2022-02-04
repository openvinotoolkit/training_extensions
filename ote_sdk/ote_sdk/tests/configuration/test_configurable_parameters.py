# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest

from ote_sdk.configuration.configurable_parameters import ConfigurableParameters
from ote_sdk.configuration.enums.config_element_type import ConfigElementType
from ote_sdk.entities.id import ID
from ote_sdk.tests.constants.ote_sdk_components import OteSdkComponent
from ote_sdk.tests.constants.requirements import Requirements


@pytest.mark.components(OteSdkComponent.OTE_SDK)
class TestConfigurableParameters:
    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_configurable_parameters(self):
        """
        <b>Description:</b>
        Check "ConfigurableParameters" class object initialization

        <b>Input data:</b>
        "ConfigurableParameters" class object with specified initialization parameters

        <b>Expected results:</b>
        Test passes if attributes of initialized "ConfigurableParameters" class object are equal to expected
        """

        def check_configurable_parameters_attributes(
            configurable_parameters: ConfigurableParameters,
            expected_header: str,
            expected_description: str,
            expected_id: ID,
            expected_visible_in_ui: bool,
        ):
            assert configurable_parameters.header == expected_header
            assert configurable_parameters.description == expected_description
            assert (
                configurable_parameters.type
                == ConfigElementType.CONFIGURABLE_PARAMETERS
            )
            assert configurable_parameters.groups == []
            assert configurable_parameters.id == expected_id
            assert configurable_parameters.visible_in_ui == expected_visible_in_ui

        header = "Test Header"
        # Checking "ConfigurableParameters" initialized with default optional parameters
        check_configurable_parameters_attributes(
            configurable_parameters=ConfigurableParameters(header=header),
            expected_header=header,
            expected_description="Default parameter group description",
            expected_id=ID(""),
            expected_visible_in_ui=True,
        )
        # Checking "ConfigurableParameters" initialized with specified optional parameters
        description = "Test Description"
        config_id = ID("Test ID")
        visible_in_ui = False
        check_configurable_parameters_attributes(
            configurable_parameters=ConfigurableParameters(
                header=header,
                description=description,
                id=config_id,
                visible_in_ui=visible_in_ui,
            ),
            expected_header=header,
            expected_description=description,
            expected_id=config_id,
            expected_visible_in_ui=visible_in_ui,
        )
