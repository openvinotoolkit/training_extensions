# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import copy

import pytest

from ote_sdk.configuration.configurable_parameters import ConfigurableParameters
from ote_sdk.configuration.elements import metadata_keys
from ote_sdk.configuration.enums import AutoHPOState
from ote_sdk.configuration.enums.config_element_type import ConfigElementType
from ote_sdk.entities.id import ID
from ote_sdk.tests.configuration.dummy_config import DatasetManagerConfig
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

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_set_metadata(self):
        """
        <b>Description:</b>
        Check "ConfigurableParameters" class parameter metadata setting

        <b>Input data:</b>
        Dummy configuration -- DatasetManagerConfig

        <b>Expected results:</b>
        Test passes if:
            1. Metadata for a given parameter inside the ConfigurableParameters can be
               set successfully,
            2. Attempting to set metadata for a non-existing parameter results in
               failure
            3. Attempting to set metadata for a non-existing metadata key results in
               failure
            4. Attempting to set metadata to a value of a type that does not match the
               original metadata item type results in failure
            5. Resetting the metadata back to its original value can be done
               successfully
        """
        # Arrange
        config = DatasetManagerConfig(
            description="Configurable parameters for the DatasetManager -- TEST ONLY",
            header="Dataset Manager configuration -- TEST ONLY",
        )
        test_parameter_name = "dummy_float_selectable"
        metadata_key = metadata_keys.AUTO_HPO_STATE
        old_value = config.get_metadata(
            test_parameter_name
        )[metadata_key]
        new_value = AutoHPOState.OPTIMIZED

        # Act
        success = config.set_metadata_value(
            parameter_name=test_parameter_name,
            metadata_key=metadata_key,
            value=new_value
        )
        no_success_invalid_param = config.set_metadata_value(
            parameter_name=test_parameter_name+'_invalid',
            metadata_key=metadata_key,
            value=new_value
        )
        no_success_invalid_key = config.set_metadata_value(
            parameter_name=test_parameter_name,
            metadata_key=metadata_key+'_invalid',
            value=new_value
        )
        no_success_invalid_value_type = config.set_metadata_value(
            parameter_name=test_parameter_name,
            metadata_key=metadata_key,
            value=str(new_value)
        )
        config_copy = copy.deepcopy(config)
        success_revert = config_copy.set_metadata_value(
            parameter_name=test_parameter_name,
            metadata_key=metadata_key,
            value=old_value
        )

        # Assert
        assert old_value != new_value
        assert all([success, success_revert])
        assert not any(
            [
                no_success_invalid_key,
                no_success_invalid_param,
                no_success_invalid_value_type
            ]
        )
        assert config.get_metadata(test_parameter_name)[metadata_key] == new_value
        assert config_copy.get_metadata(test_parameter_name)[metadata_key] == old_value
