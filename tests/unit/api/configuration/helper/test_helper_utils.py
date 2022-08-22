# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from enum import Enum
from pathlib import Path

import pytest
import yaml
from omegaconf import DictConfig

from otx.api.configuration.helper.utils import (
    _search_in_config_dict_inner,
    deserialize_enum_value,
    ids_to_strings,
    input_to_config_dict,
    search_in_config_dict,
)
from otx.api.entities.id import ID
from tests.unit.api.constants.components import OtxSdkComponent
from tests.unit.api.constants.requirements import Requirements


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestUtilsFunctions:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_utils_search_in_config_dict_inner(self):
        """
        <b>Description:</b>
        Check "_search_in_config_dict_inner" function

        <b>Input data:</b>
        "config_dict" dictionary, "key_to_search" string, "prior_keys" list, "results" list

        <b>Expected results:</b>
        Test passes if list returned by "_search_in_config_dict_inner" function is equal to expected

        <b>Steps</b>
        1. Check list returned by "_search_in_config_dict_inner" function with default values of optional parameters
        2. Check list returned by "_search_in_config_dict_inner" function with specified "prior_keys" parameter
        3. Check list returned by "_search_in_config_dict_inner" function with specified "results" parameter
        4. Check list returned by "_search_in_config_dict_inner" function with specified both optional parameters
        5. Check list returned by "_search_in_config_dict_inner" function with list object specified as "config_dict"
        parameter
        """
        # Checking list returned by "_search_in_config_dict_inner" with default values of optional parameters
        config_dict = {"key_1": 2, "key_2": 4, "key_3": 8}
        # Checking search of existing key
        assert _search_in_config_dict_inner(config_dict=config_dict, key_to_search="key_2") == [(4, [])]
        # Checking search of non-existing key
        assert _search_in_config_dict_inner(config_dict=config_dict, key_to_search="key_4") == []

        # Checking list returned by "_search_in_config_dict_inner" with specified "prior_keys"
        prior_keys = ["prior_key_1", "prior_key_2"]
        # Checking search of existing key
        assert _search_in_config_dict_inner(config_dict=config_dict, key_to_search="key_2", prior_keys=prior_keys) == [
            (4, prior_keys)
        ]
        # Checking search of non-existing key
        assert _search_in_config_dict_inner(config_dict=config_dict, key_to_search="key_4", prior_keys=prior_keys) == []
        # Checking list returned by "_search_in_config_dict_inner" with specified "results"
        # Checking search of existing key
        results = [(4, ["result_key", "other_result_key"])]
        expected_results = [(4, ["result_key", "other_result_key"]), (2, [])]
        assert (
            _search_in_config_dict_inner(config_dict=config_dict, key_to_search="key_1", results=results)
            == expected_results
        )
        assert results == expected_results
        # Checking search of non-existing key
        results = [(4, ["result_key", "other_result_key"])]
        expected_results = list(results)
        assert (
            _search_in_config_dict_inner(config_dict=config_dict, key_to_search="key_4", results=results)
            == expected_results
        )
        assert results == expected_results
        # Checking list returned by "_search_in_config_dict_inner" with specified "prior_keys" and "results"
        # Checking search of existing key
        results = [(4, ["result_key", "other_result_key"])]
        expected_results = [
            (4, ["result_key", "other_result_key"]),
            (8, ["prior_key_1", "prior_key_2"]),
        ]
        assert (
            _search_in_config_dict_inner(
                config_dict=config_dict,
                key_to_search="key_3",
                prior_keys=prior_keys,
                results=results,
            )
            == expected_results
        )
        assert results == expected_results
        # Checking search of non-existing key
        results = [(4, ["result_key", "other_result_key"])]
        expected_results = list(results)
        assert (
            _search_in_config_dict_inner(
                config_dict=config_dict,
                key_to_search="key_4",
                prior_keys=prior_keys,
                results=results,
            )
            == expected_results
        )
        assert results == expected_results
        # Checking list returned by "_search_in_config_dict_inner" with list specified as "config_dict"
        # Checking search of existing key
        config_list = [1, 3, 9]
        results = [(4, ["result_key", "other_result_key"])]
        expected_results = [
            (4, ["result_key", "other_result_key"]),
            (3, ["prior_key_1", "prior_key_2"]),
        ]
        assert (
            _search_in_config_dict_inner(
                config_dict=config_list,  # type: ignore
                key_to_search=1,  # type: ignore
                prior_keys=prior_keys,
                results=results,
            )
            == expected_results
        )
        assert results == expected_results
        # Checking search of non-existing key
        results = [(4, ["result_key", "other_result_key"])]
        expected_results = list(results)
        assert (
            _search_in_config_dict_inner(
                config_dict=config_list,  # type: ignore
                key_to_search=5,  # type: ignore
                prior_keys=prior_keys,
                results=results,
            )
            == expected_results
        )
        assert results == expected_results

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_utils_search_in_config_dict(self):
        """
        <b>Description:</b>
        Check "search_in_config_dict" function

        <b>Input data:</b>
        "config_dict" dictionary, "key_to_search" string

        <b>Expected results:</b>
        Test passes if list returned by "search_in_config_dict" function is equal to expected

        <b>Steps</b>
        1. Check list returned by "search_in_config_dict" function when searching key in "config_dict" dictionary
        2. Check list returned by "search_in_config_dict" function when searching key in "config_dict" list
        """
        # Checking list returned by "search_in_config_dict" when searching key in dictionary "config_dict"
        config_dict = {"key_1": 2, "key_2": 4, "key_3": 8}
        assert search_in_config_dict(config_dict, "key_1") == [(2, [])]
        # Checking search of non-existing key
        assert search_in_config_dict(config_dict, "key_4") == []
        # Checking list returned by "search_in_config_dict" when searching key in list "config_dict"
        config_dict = [1, 3, 9]
        assert search_in_config_dict(config_dict, 1) == [(3, [])]  # type: ignore
        # Checking search of non-existing key
        assert search_in_config_dict(config_dict, "key_1") == []  # type: ignore

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_utils_input_to_config_dict(self):
        """
        <b>Description:</b>
        Check "input_to_config_dict" function

        <b>Input data:</b>
        "input_config" string, DictConfig or dictionary object, "check_config_type" bool value

        <b>Expected results:</b>
        Test passes if dictionary returned by "input_to_config_dict" function is equal to expected

        <b>Steps</b>
        1. Check dictionary returned by "input_to_config_dict" function when "check_config_type" parameter is "True"
        2. Check dictionary returned by "input_to_config_dict" function when "check_config_type" parameter is "False"
        and "type" key is not specified in "input_config" parameter
        3. Check that ValueError exception is raised when type of "input_config" parameter is not equal to string,
        DictConfig or dictionary
        4. Check that ValueError exception is raised by "input_to_config_dict" function when "check_config_type"
        parameter is "True" and "type" key is not specified in "input_config" parameter
        5. Check that ValueError exception is raised by "input_to_config_dict" function when "check_config_type"
        parameter is "True" and "type" key in "input_config" parameter is equal to unexpected value
        """
        # Checking dictionary returned by "input_to_config_dict" when "check_config_type" is "True"
        path_to_config = str(Path(__file__).parent / Path(r"../dummy_config.yaml"))
        with open(path_to_config, "r", encoding="UTF-8") as file:
            expected_path_to_config_dict = yaml.safe_load(file)
        string_config = "{'str_key_1': 2, 'str_key_2': 4, 'str_key_3': 8, 'type': PARAMETER_GROUP}"
        expected_string_config_dict = {
            "str_key_1": 2,
            "str_key_2": 4,
            "str_key_3": 8,
            "type": "PARAMETER_GROUP",
        }
        dict_config = {
            "dict_key_1": 2,
            "dict_key_2": 4,
            "dict_key_3": 8,
            "type": "PARAMETER_GROUP",
        }
        expected_dict_config = {
            "dict_key_1": 2,
            "dict_key_2": 4,
            "dict_key_3": 8,
            "type": "PARAMETER_GROUP",
        }
        dict_config_instance = DictConfig(
            content={
                "DictConfig_key_1": 2,
                "DictConfig_key_2": 4,
                "DictConfig_key_3": 8,
                "type": "PARAMETER_GROUP",
            }
        )
        expected_dict_config_instance_dict = {
            "DictConfig_key_1": 2,
            "DictConfig_key_2": 4,
            "DictConfig_key_3": 8,
            "type": "PARAMETER_GROUP",
        }

        for input_config, expected_dict in [
            (path_to_config, expected_path_to_config_dict),
            (string_config, expected_string_config_dict),
            (dict_config, expected_dict_config),
            (dict_config_instance, expected_dict_config_instance_dict),
        ]:
            assert input_to_config_dict(input_config=input_config, check_config_type=True) == expected_dict
        # Checking dictionary returned by "input_to_config_dict" when "check_config_type" is "False" and "type" key
        # is not specified in "input_config"
        string_config = "{'str_key_1': 2, 'str_key_2': 4, 'str_key_3': 8}"
        expected_string_config_dict = {"str_key_1": 2, "str_key_2": 4, "str_key_3": 8}
        dict_config = {"dict_key_1": 2, "dict_key_2": 4, "dict_key_3": 8}
        expected_dict_config = {"dict_key_1": 2, "dict_key_2": 4, "dict_key_3": 8}
        dict_config_instance = DictConfig(
            content={
                "DictConfig_key_1": 2,
                "DictConfig_key_2": 4,
                "DictConfig_key_3": 8,
            }
        )
        expected_dict_config_instance_dict = {
            "DictConfig_key_1": 2,
            "DictConfig_key_2": 4,
            "DictConfig_key_3": 8,
        }

        for input_config, expected_dict in [
            (path_to_config, expected_path_to_config_dict),
            (string_config, expected_string_config_dict),
            (dict_config, expected_dict_config),
            (dict_config_instance, expected_dict_config_instance_dict),
        ]:
            assert input_to_config_dict(input_config=input_config, check_config_type=False) == expected_dict
        # Checking that ValueError exception is raised when type of "input_config" is not equal to string, DictConfig or
        # dictionary
        with pytest.raises(ValueError):
            input_to_config_dict(input_config=1)  # type: ignore
        # Checking that ValueError exception is raised by "input_to_config_dict" when "check_config_type" is "True" and
        # "type" key is not specified in "input_config"
        for none_type_input_config in [
            "{'key_1': 2, 'key_2': 4, 'key_3': 8}",
            {"key_1": 2, "key_2": 4, "key_3": 8},
            DictConfig(content={"key_1": 2, "key_2": 4, "key_3": 8}),
        ]:
            with pytest.raises(ValueError):
                input_to_config_dict(input_config=none_type_input_config)
        # Check that ValueError exception is raised by "input_to_config_dict" when "check_config_type" is "True" and
        # "type" key in "input_config" is equal to unexpected value
        for none_type_input_config in [
            "{'key_1': 2, 'key_2': 4, 'key_3': 8, 'type': unexpected_type}",
            {"key_1": 2, "key_2": 4, "key_3": 8, "type": "unexpected_type"},
            DictConfig(content={"key_1": 2, "key_2": 4, "key_3": 8, "type": "unexpected_type"}),
        ]:
            with pytest.raises(ValueError):
                input_to_config_dict(input_config=none_type_input_config)

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_utils_deserialize_enum_value(self):
        """
        <b>Description:</b>
        Check "deserialize_enum_value" function

        <b>Input data:</b>
        "enum_type" Enum object, "value" Enum element or string object

        <b>Expected results:</b>
        Test passes if value returned by "deserialize_enum_value" function is equal to expected

        <b>Steps</b>
        1. Check value returned by "deserialize_enum_value" function when Enum class element is specified as "value"
        parameter
        2. Check value returned by "deserialize_enum_value" function when string is specified as "value" parameter
        3. Check that ValueError exception is raised when type of "value" parameter for "deserialize_enum_value"
        function is not equal to Enum or string
        """

        class ValidationEnum(Enum):
            FIRST = "first element"
            SECOND = "second element"

        # Checking value returned by "deserialize_enum_value" when Enum class element is specified as "value"
        assert deserialize_enum_value(value=ValidationEnum.FIRST, enum_type=ValidationEnum) == ValidationEnum.FIRST
        # Checking value returned by "deserialize_enum_value" when string is specified as "value"
        assert deserialize_enum_value(value="SECOND", enum_type=ValidationEnum) == ValidationEnum.SECOND
        with pytest.raises(KeyError):
            deserialize_enum_value(value="THIRD", enum_type=ValidationEnum)
        # Checking that ValueError exception is raised when type of "value" for "deserialize_enum_value" is not equal to
        # Enum or string
        with pytest.raises(ValueError):
            deserialize_enum_value(value=1, enum_type=ValidationEnum)  # type: ignore

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_utils_ids_to_strings(self):
        """
        <b>Description:</b>
        Check "ids_to_strings" function

        <b>Input data:</b>
        "config_dict" dictionary

        <b>Expected results:</b>
        Test passes if dictionary returned by "ids_to_strings" function is equal to expected
        """
        config_dict = {
            "id_key_1": ID("1"),
            "not_id_key_1": 2,
            "not_id_key_2": 3,
            "id_key_2": ID("4"),
        }
        expected_dict = {
            "id_key_1": "1",
            "not_id_key_1": 2,
            "not_id_key_2": 3,
            "id_key_2": "4",
        }
        assert ids_to_strings(config_dict) == expected_dict
        assert config_dict == expected_dict
