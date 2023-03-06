# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from otx.api.configuration import cfg_helper
from otx.api.configuration.elements import metadata_keys
from otx.api.configuration.enums import AutoHPOState, ModelLifecycle
from tests.unit.api.configuration.dummy_config import (
    DatasetManagerConfig,
    SomeEnumSelectable,
)
from tests.unit.api.constants.components import OtxSdkComponent
from tests.unit.api.constants.requirements import Requirements


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestConfigurationHelper:
    @staticmethod
    def __get_path_to_file(filename: str):
        """
        Return the path to the file named 'filename', which lives in the tests/configuration directory
        """
        return str(Path(__file__).parent / Path(filename))

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_config_reconstruction(self):
        """
        <b>Description:</b>
        This test verifies that a configuration can be converted to a dictionary and yaml string, and back again. The
        test passes if the reconstructed configuration is equal to the initial one.

        <b>Input data:</b>
        Dummy configuration -- DatasetManagerConfig

        <b>Expected results:</b>
        Test passes if reconstructed configuration is equal to the original one

        <b>Steps</b>
        1. Create configuration
        2. Convert to dictionary and yaml string representation
        3. Reconstruct configuration from dict
        4. Reconstruct configuration from yaml string
        5. Verify that contents of reconstructed configs are equal to original config
        """
        # Initialize the config object
        config = DatasetManagerConfig(
            description="Configurable parameters for the DatasetManager -- TEST ONLY",
            header="Dataset Manager configuration -- TEST ONLY",
        )

        # Convert config to dictionary and to yaml string
        cfg = cfg_helper.convert(config, dict)
        cfg_yaml = cfg_helper.convert(config, str)

        # Reconstruct the config from dictionary and from yaml string
        reconstructed_config = cfg_helper.create(cfg)
        reconstructed_config_from_yaml = cfg_helper.create(cfg_yaml)

        # Compare the config dictionaries. Order of some parameters may change in the conversion, so dictionary
        # comparison will work while comparing objects or yaml strings directly likely does not result in equality.
        assert cfg == cfg_helper.convert(reconstructed_config, dict)
        assert cfg == cfg_helper.convert(reconstructed_config_from_yaml, dict)

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_config_reconstruction_with_metadata_change(self):
        """
        <b>Description:</b>
        This test verifies that metadata changes are capture upon converting a
        configuration to a dictionary and yaml string, and back again. The test passes
        if the reconstructed configuration is equal to the initial one, with
        non-default metadata values set.

        <b>Input data:</b>
        Dummy configuration -- DatasetManagerConfig

        <b>Expected results:</b>
        Test passes if metadata changes are reflected in the reconstructed
        configuration, such that it is equal to the original one

        <b>Steps</b>
        1. Create configuration
        2. Set metadata for a parameter to a non-default value
        3. Convert to dictionary and yaml string representation
        4. Reconstruct configuration from dict
        5. Reconstruct configuration from yaml string
        6. Verify that contents of reconstructed configs are equal to original config
        """
        # Arrange
        # Initialize the config object
        config = DatasetManagerConfig(
            description="Configurable parameters for the DatasetManager -- TEST ONLY",
            header="Dataset Manager configuration -- TEST ONLY",
        )
        test_parameter_name = "dummy_float_selectable"
        metadata_key = metadata_keys.AUTO_HPO_STATE
        old_state = config.get_metadata(test_parameter_name)[metadata_key]
        new_state = AutoHPOState.OPTIMIZED
        set_success = config.set_metadata_value(
            parameter_name=test_parameter_name,
            metadata_key=metadata_key,
            value=new_state,
        )

        # Act
        # Convert config to dictionary and to yaml string
        cfg = cfg_helper.convert(config, dict)
        cfg_yaml = cfg_helper.convert(config, str)

        # Reconstruct the config from dictionary and from yaml string
        reconstructed_config = cfg_helper.create(cfg)
        reconstructed_config_from_yaml = cfg_helper.create(cfg_yaml)

        # Assert
        assert old_state != new_state
        assert set_success
        # Check that metadata changes are properly converted
        assert reconstructed_config.get_metadata(parameter_name=test_parameter_name)[metadata_key] == new_state
        assert (
            reconstructed_config_from_yaml.get_metadata(parameter_name=test_parameter_name)[metadata_key] == new_state
        )
        # Compare the config dictionaries
        assert cfg == cfg_helper.convert(reconstructed_config, dict)
        assert cfg == cfg_helper.convert(reconstructed_config_from_yaml, dict)

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_creation_from_yaml(self):
        """
        <b>Description:</b>
        This test verifies that a configuration can be created from a yaml file

        <b>Input data:</b>
        dummy_config.py   -- DatasetManagerConfig class definition
        dummy_config.yaml -- yaml file specifying a configuration equivalent to DatasetManagerConfig, but in yaml

        <b>Expected results:</b>
        Test passes if the contents of the config created from dummy_config.yaml is equal to the configuration in
        DatasetManagerConfig

        <b>Steps</b>
        1. Create configuration from class
        2. Create configuration from yaml
        3. Convert both configs to dictionary
        4. Compare resulting dictionaries
        """
        # Initialize the config
        config = DatasetManagerConfig(
            description="Configurable parameters for the DatasetManager -- TEST ONLY",
            header="Dataset Manager configuration -- TEST ONLY",
        )

        cfg_from_yaml = cfg_helper.create(self.__get_path_to_file("./dummy_config.yaml"))

        # Compare the config dictionaries. Order of some parameters may change in the conversion, so dictionary
        # comparison will work while comparing objects or yaml strings directly likely does not result in equality.
        cfg_dict = cfg_helper.convert(config, dict)
        cfg_from_yaml_dict = cfg_helper.convert(cfg_from_yaml, dict)

        # Check the parameter groups individually, to narrow down any errors more easily
        cfg_subset_parameters = cfg_dict.pop("subset_parameters")
        yamlcfg_subset_parameters = cfg_from_yaml_dict.pop("subset_parameters")
        assert cfg_subset_parameters == yamlcfg_subset_parameters

        cfg_nested_group = cfg_dict.pop("nested_parameter_group")
        yamlcfg_nested_group = cfg_from_yaml_dict.pop("nested_parameter_group")
        assert cfg_nested_group == yamlcfg_nested_group

        assert config.dummy_selectable == cfg_dict["dummy_selectable"]["value"]

        assert cfg_dict == cfg_from_yaml_dict

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_broken_config(self):
        """
        <b>Description:</b>
        This test verifies that a configuration created from a yaml file that contains invalid parameter values will
        raise a ValueError. It also verifies that no ValueError is raised if the config is created with valid values.

        <b>Input data:</b>
        dummy_broken_config.yaml -- yaml file specifying a configuration, holding an out-of-bounds value for the number
                                    of epochs

        <b>Expected results:</b>
        Test passes if a ValueError is raised upon creating the config from yaml, and no ValueError is raised upon
        config creation after correcting the out-of-bounds value.

        <b>Steps</b>
        1. Create configuration from dummy_broken_config.yaml
        2. Assert that ValueError is raised
        3. Load dummy_broken_config.yaml as dictionary and correct invalid epochs value
        4. Create configuration from the corrected dictionary
        5. Assert that epochs value is set in the configuration correctly
        """
        # Loading a config that has an invalid value (-5 for epochs) should raise a ValueError due to runtime
        # input validation upon config creation.
        broken_config_path = self.__get_path_to_file("dummy_broken_config.yaml")
        with pytest.raises(ValueError) as error:
            config = cfg_helper.create(broken_config_path)

        assert "Invalid value set for epochs: -5 is out of bounds." == str(error.value)

        # Test correcting the broken config by first loading it from the yaml file, and then setting epochs to a valid
        # value. Config should now be created correctly. Finally, assert that the value for epochs has been corrected
        dict_config = OmegaConf.load(broken_config_path)
        dict_config.learning_parameters.epochs.value = 10
        config = cfg_helper.create(dict_config)
        assert config.learning_parameters.epochs == 10

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_validation(self):
        """
        <b>Description:</b>
        This test verifies that validating the parameter values in the configuration works. Validation should raise a
        ValueError if any of the config parameters is set to an invalid value.

        <b>Input data:</b>
        dummy_config.py   -- DatasetManagerConfig class definition


        <b>Expected results:</b>
        Test passes if a ValueError is raised upon validation of the configuration, if it contains invalid values for
        any of its parameters.

        <b>Steps</b>
        1. Create DatasetManagerConfig configuration
        2. Set test_proportion to an out of bounds value
        3. Validate config and assert that ValueError is raised
        4. Set test_proportion to a valid value
        5. Validate config and assert that no ValueError is raised
        6. Set dummy_selectable parameter to invalid value
        7. Validate config and assert that ValueError is raised
        8. Correct dummy_selectable and assert that ValueError is not raised upon validation
        """
        # Initialize the config
        config = DatasetManagerConfig(
            description="TEST ONLY",
            header="TEST ONLY",
        )

        # Assert that config passes validation initially
        assert cfg_helper.validate(config)

        # Set invalid test_proportion, and assert that this raises an error upon validation
        with pytest.raises(ValueError):
            config.subset_parameters.test_proportion = 1.1

        # Assert that validation passes again after restoring a value that is within the bounds
        config.subset_parameters.test_proportion = 0.25
        assert cfg_helper.validate(config)

        # Set value that is not one of the options for dummy_selectable, assert that this raises a ValueError
        with pytest.raises(ValueError):
            config.dummy_selectable = "invalid_value"

        # Assert that validation passes again after restoring to a value that is in the options list
        config.dummy_selectable = "option_c"
        assert cfg_helper.validate(config)

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_substitute_values(self):
        """
        <b>Description:</b>
        This test verifies that parameter values can be substituted into a configuration, using an input dictionary or
        input configuration.

        <b>Input data:</b>
        dummy_config.py   -- DatasetManagerConfig class definition

        <b>Expected results:</b>
        Test passes if the configuration is successfully converted into a dictionary,
        with keys and values matching the structure of the original configuration

        <b>Steps</b>
        1. Create configuration from dummy_config.yaml
        2. Check that values are set according to their initial definition in dummy_config.yaml
        3. Load dummy_config.yaml as an input dictionary, and change some of its parameter values.
        4. Substitute the values from the dictionary
        5. Check that the parameter values have been changed correctly in the config
        6. Change the parameter values in the input dictionary again
        7. Convert the input dictionary to an input configuration
        8. Substitute the values from the input configuration
        9. Check that the parameter values have been changed correctly in the config
        """
        # Initialize the config from yaml
        config = cfg_helper.create(self.__get_path_to_file("dummy_config.yaml"))

        # Assert that the values are set according to what is specified in the yaml
        assert config.subset_parameters.test_proportion == 0.15
        assert config.number_of_samples_for_auto_train == 5

        # Load the config as a dict from the yaml and change some of the values
        config_dict = OmegaConf.load(self.__get_path_to_file("dummy_config.yaml"))
        config_dict.subset_parameters.test_proportion.value = 0.05
        config_dict.number_of_samples_for_auto_train.value = 50

        # Substitute values from this dict
        cfg_helper.substitute_values(config, value_input=config_dict)

        # Assert that the values are changed in the config, according to what was substituted above
        assert config.subset_parameters.test_proportion == 0.05
        assert config.number_of_samples_for_auto_train == 50

        # Convert config_dict to actual config object, and then use that as input for the value substitution
        config_dict.subset_parameters.test_proportion.value = 0.80
        config_dict.number_of_samples_for_auto_train.value = 500
        reconstructed_config = cfg_helper.create(config_dict)
        cfg_helper.substitute_values(config, value_input=reconstructed_config)

        # Assert that the values are changed in the config, according to what was substituted above
        assert config.subset_parameters.test_proportion == 0.80
        assert config.number_of_samples_for_auto_train == 500

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_substitute_values_for_lifecycle(self):
        """
        <b>Description:</b>
        This test verifies that parameter values can be substituted into a
        configuration, conditional on the phase in the model lifecycle that they affect

        <b>Input data:</b>
        dummy_config.py   -- DatasetManagerConfig class definition

        <b>Expected results:</b>
        Test passes if the configuration values that affect the model life cycle
        specified are updated, whereas others are not

        <b>Steps</b>
        1. Create configuration from dummy_config.yaml
        2. Check that values are set according to their initial definition in
            dummy_config.yaml
        3. Create second config from dummy_config.yaml, and change some of its values
        4. Substitute the values from the modified config
        5. Check that the parameter values have been changed correctly in the config
        6. Check that parameter values for parameters that do not belong to the target
            model lifecycle did not change
        """
        # Initialize the config from yaml
        config = cfg_helper.create(self.__get_path_to_file("dummy_config.yaml"))

        # Assert that the values are set according to what is specified in the yaml
        assert config.subset_parameters.test_proportion == 0.15
        assert config.subset_parameters.get_metadata("test_proportion")["affects_outcome_of"] == ModelLifecycle.TRAINING
        assert config.dummy_float_selectable == 2
        assert config.get_metadata("dummy_float_selectable")["affects_outcome_of"] == ModelLifecycle.NONE
        assert config.dummy_selectable == SomeEnumSelectable.BOGUS_NAME
        assert config.get_metadata("dummy_selectable")["affects_outcome_of"] == ModelLifecycle.INFERENCE

        # Load the config again from the yaml and change some of the values
        config_2 = cfg_helper.create(self.__get_path_to_file("dummy_config.yaml"))
        config_2.subset_parameters.test_proportion = 0.05
        config_2.dummy_float_selectable = 4.0
        config_2.dummy_selectable = SomeEnumSelectable.TEST_NAME1

        cfg_helper.substitute_values_for_lifecycle(config, config_2, model_lifecycle=ModelLifecycle.INFERENCE)

        assert config.subset_parameters.test_proportion == 0.15
        assert config.dummy_selectable == SomeEnumSelectable.TEST_NAME1
        assert config.dummy_float_selectable == 2

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_substitute_values_for_lifecycle_list(self):
        """
        <b>Description:</b>
        This test verifies that parameter values can be substituted into a
        configuration, conditional on the phase in the model lifecycle that they affect

        <b>Input data:</b>
        dummy_config.py   -- DatasetManagerConfig class definition

        <b>Expected results:</b>
        Test passes if the configuration values that affect the model life cycle
        specified are updated, whereas others are not

        <b>Steps</b>
        1. Create configuration from dummy_config.yaml
        2. Check that values are set according to their initial definition in
            dummy_config.yaml
        3. Create second config from dummy_config.yaml, and change some of its values
        4. Substitute the values from the modified config
        5. Check that the parameter values have been changed correctly in the config
        6. Check that parameter values for parameters that do not belong to the target
            model lifecycle did not change
        """
        # Initialize the config from yaml
        config = cfg_helper.create(self.__get_path_to_file("dummy_config.yaml"))

        # Assert that the values are set according to what is specified in the yaml
        assert config.subset_parameters.test_proportion == 0.15
        assert config.subset_parameters.get_metadata("test_proportion")["affects_outcome_of"] == ModelLifecycle.TRAINING
        assert config.dummy_float_selectable == 2
        assert config.get_metadata("dummy_float_selectable")["affects_outcome_of"] == ModelLifecycle.NONE
        assert config.dummy_selectable == SomeEnumSelectable.BOGUS_NAME
        assert config.get_metadata("dummy_selectable")["affects_outcome_of"] == ModelLifecycle.INFERENCE

        # Load the config again from the yaml and change some of the values
        config_2 = cfg_helper.create(self.__get_path_to_file("dummy_config.yaml"))
        config_2.subset_parameters.test_proportion = 0.05
        config_2.dummy_float_selectable = 4.0
        config_2.dummy_selectable = SomeEnumSelectable.TEST_NAME1

        cfg_helper.substitute_values_for_lifecycle(
            config,
            config_2,
            model_lifecycle=[ModelLifecycle.INFERENCE, ModelLifecycle.NONE],
        )

        assert config.subset_parameters.test_proportion == 0.15
        assert config.dummy_selectable == SomeEnumSelectable.TEST_NAME1
        assert config.dummy_float_selectable == 4

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_values_only_conversion(self):
        """
        <b>Description:</b>
        This test verifies that converting a configuration into a dictionary while
        retaining only parameter values (and discarding metadata) works

        <b>Input data:</b>
        dummy_config.yaml -- yaml file specifying a configuration equivalent to
        DatasetManagerConfig, but in yaml

        <b>Expected results:</b>
        Test passes if the parameter values of the configuration are updated according to the input dictionary and
        input configuration from which the values are substituted.

        <b>Steps</b>
        1. Create DatasetManagerConfig configuration
        2. Change a couple of parameters away from their defaults
        3. Convert the configuration to a dictionary, with `values_only=True` to
            discard meta-data
        4. Assert that the resulting dictionary contains the expected values
        """
        config = DatasetManagerConfig()
        config.subset_parameters.test_proportion = 0.3
        config.dummy_selectable = SomeEnumSelectable.OPTION_C

        config_dict = cfg_helper.convert(config, target=dict, values_only=True)

        assert config_dict["subset_parameters"]["test_proportion"] == 0.3
        assert config_dict["dummy_selectable"] == SomeEnumSelectable.OPTION_C
        assert config_dict["number_of_samples_for_auto_train"] == 5
        assert config_dict["nested_parameter_group"]["subgroup_one"]["bogus_parameter_one"] == 42
