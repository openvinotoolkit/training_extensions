"""
Utils for checking functions and methods arguments
"""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import typing
from abc import ABC, abstractmethod
from os.path import exists

import yaml
from numpy import floating
from omegaconf import DictConfig


def raise_value_error_if_parameter_has_unexpected_type(
    parameter, parameter_name, expected_type
):
    """Function raises ValueError exception if parameter has unexpected type"""
    if expected_type == float:
        expected_type = (int, float, floating)
    if not isinstance(parameter, expected_type):
        parameter_type = type(parameter)
        raise ValueError(
            f"Unexpected type of '{parameter_name}' parameter, expected: {expected_type}, actual: {parameter_type}"
        )


def check_nested_elements_type(iterable, parameter_name, expected_type):
    """Function raises ValueError exception if one of elements in collection has unexpected type"""
    for element in iterable:
        check_parameter_type(
            parameter=element,
            parameter_name=f"nested {parameter_name}",
            expected_type=expected_type,
        )


def check_dictionary_keys_values_type(
    parameter, parameter_name, expected_key_class, expected_value_class
):
    """Function raises ValueError exception if dictionary key or value has unexpected type"""
    for key, value in parameter.items():
        check_parameter_type(
            parameter=key,
            parameter_name=f"key in {parameter_name}",
            expected_type=expected_key_class,
        )
        check_parameter_type(
            parameter=value,
            parameter_name=f"value in {parameter_name}",
            expected_type=expected_value_class,
        )


def check_parameter_type(parameter, parameter_name, expected_type):
    """Function extracts nested expected types and raises ValueError exception if parameter has unexpected type"""
    # pylint: disable=W0212
    if not isinstance(expected_type, typing._GenericAlias):  # type: ignore
        raise_value_error_if_parameter_has_unexpected_type(
            parameter=parameter,
            parameter_name=parameter_name,
            expected_type=expected_type,
        )
        return
    if expected_type == typing.Any:
        return
    origin_class = expected_type.__dict__.get("__origin__")
    # Checking origin class
    raise_value_error_if_parameter_has_unexpected_type(
        parameter=parameter,
        parameter_name=parameter_name,
        expected_type=origin_class,
    )
    # Checking nested elements
    args = expected_type.__dict__.get("__args__")
    if issubclass(origin_class, typing.Sequence) and args:
        if len(args) != 1:
            raise TypeError(
                "length of nested expected types for Sequence should be equal to 1"
            )
        check_nested_elements_type(
            iterable=parameter,
            parameter_name=parameter_name,
            expected_type=args,
        )
    elif origin_class == dict and args:
        if len(args) != 2:
            raise TypeError(
                "length of nested expected types for dictionary should be equal to 2"
            )
        key, value = args
        check_dictionary_keys_values_type(
            parameter=parameter,
            parameter_name=parameter_name,
            expected_key_class=key,
            expected_value_class=value,
        )


class BaseInputArgumentChecker(ABC):
    """Abstract class to check input arguments"""

    @abstractmethod
    def check(self):
        """Abstract method to check input arguments"""
        raise NotImplementedError("The check is not implemented")


def check_input_param_type(*checks: BaseInputArgumentChecker):
    """Function to apply methods on checks according to their type"""
    for param_check in checks:
        if not isinstance(param_check, BaseInputArgumentChecker):
            raise TypeError(f"Wrong parameter of check_input_param: {param_check}")
        param_check.check()


class RequiredParamTypeCheck(BaseInputArgumentChecker):
    """Class to check required input parameters"""

    def __init__(self, parameter, parameter_name, expected_type):
        self.parameter = parameter
        self.parameter_name = parameter_name
        self.expected_type = expected_type

    def check(self):
        """Method raises ValueError exception if required parameter has unexpected type"""
        check_parameter_type(
            parameter=self.parameter,
            parameter_name=self.parameter_name,
            expected_type=self.expected_type,
        )


class OptionalParamTypeCheck(RequiredParamTypeCheck):
    """Class to check optional input parameters"""

    def check(self):
        """Method checks if optional parameter exists and raises ValueError exception if it has unexpected type"""
        if self.parameter is not None:
            check_parameter_type(
                parameter=self.parameter,
                parameter_name=self.parameter_name,
                expected_type=self.expected_type,
            )


def check_file_extension(
    file_path: str, file_path_name: str, expected_extensions: list
):
    """Function raises ValueError exception if file has unexpected extension"""
    file_extension = file_path.split(".")[-1].lower()
    if file_extension not in expected_extensions:
        raise ValueError(
            f"Unexpected extension of {file_path_name} file. expected: {expected_extensions} actual: {file_extension}"
        )


def check_that_null_character_absents_in_string(parameter: str, parameter_name: str):
    """Function raises ValueError exception if null character: '\0' is specified in path to file"""
    if "\0" in parameter:
        raise ValueError(f"\\0 is specified in {parameter_name}: {parameter}")


def check_that_file_exists(file_path: str, file_path_name: str):
    """Function raises ValueError exception if file not exists"""
    if not exists(file_path):
        raise ValueError(
            f"File {file_path} specified in '{file_path_name}' parameter not exists"
        )


def check_that_parameter_is_not_empty(parameter, parameter_name):
    """Function raises ValueError if parameter is empty"""
    if not parameter:
        raise ValueError(f"parameter {parameter_name} is empty")


def check_that_all_characters_printable(parameter, parameter_name, allow_crlf=False):
    """Function raises ValueError if one of string-parameter characters is not printable"""
    if not allow_crlf:
        all_characters_printable = all(c.isprintable() for c in parameter)
    else:
        all_characters_printable = all(
            (c.isprintable() or c == "\n" or c == "\r") for c in parameter
        )
    if not all_characters_printable:
        raise ValueError(
            fr"parameter {parameter_name} has not printable symbols: {parameter}"
        )


class InputConfigCheck(BaseInputArgumentChecker):
    """Class to check input config_parameters"""

    def __init__(self, parameter):
        self.parameter = parameter

    def check(self):
        """Method raises ValueError exception if "input_config" parameter is not equal to expected"""
        parameter_name = "input_config"
        raise_value_error_if_parameter_has_unexpected_type(
            parameter=self.parameter,
            parameter_name=parameter_name,
            expected_type=(str, DictConfig, dict),
        )
        check_that_parameter_is_not_empty(
            parameter=self.parameter, parameter_name=parameter_name
        )
        if isinstance(self.parameter, str):
            check_that_null_character_absents_in_string(
                parameter=self.parameter, parameter_name=parameter_name
            )
            # yaml-format string is specified
            if isinstance(yaml.safe_load(self.parameter), dict):
                check_that_all_characters_printable(
                    parameter=self.parameter,
                    parameter_name=parameter_name,
                    allow_crlf=True,
                )
            # Path to file is specified
            else:
                check_file_extension(
                    file_path=self.parameter,
                    file_path_name=parameter_name,
                    expected_extensions=["yaml"],
                )
                check_that_all_characters_printable(
                    parameter=self.parameter, parameter_name=parameter_name
                )
                check_that_file_exists(
                    file_path=self.parameter, file_path_name=parameter_name
                )


class FilePathCheck(BaseInputArgumentChecker):
    """Class to check file_path-like parameters"""

    def __init__(self, parameter, parameter_name, expected_file_extension):
        self.parameter = parameter
        self.parameter_name = parameter_name
        self.expected_file_extensions = expected_file_extension

    def check(self):
        """Method raises ValueError exception if file path parameter is not equal to expected"""
        raise_value_error_if_parameter_has_unexpected_type(
            parameter=self.parameter,
            parameter_name=self.parameter_name,
            expected_type=str,
        )
        check_that_parameter_is_not_empty(
            parameter=self.parameter, parameter_name=self.parameter_name
        )
        check_file_extension(
            file_path=self.parameter,
            file_path_name=self.parameter_name,
            expected_extensions=self.expected_file_extensions,
        )
        check_that_null_character_absents_in_string(
            parameter=self.parameter, parameter_name=self.parameter_name
        )
        check_that_all_characters_printable(
            parameter=self.parameter, parameter_name=self.parameter_name
        )
        check_that_file_exists(
            file_path=self.parameter, file_path_name=self.parameter_name
        )


def check_is_parameter_like_dataset(parameter, parameter_name):
    """Function raises ValueError exception if parameter does not have __len__, __getitem__ and get_subset attributes of
    DataSet-type object"""
    for expected_attribute in ("__len__", "__getitem__", "get_subset"):
        if not hasattr(parameter, expected_attribute):
            parameter_type = type(parameter)
            raise ValueError(
                f"parameter {parameter_name} has type {parameter_type} which does not have expected "
                f"'{expected_attribute}' dataset attribute"
            )


class DatasetParamTypeCheck(BaseInputArgumentChecker):
    """Class to check DataSet-like parameters"""

    def __init__(self, parameter, parameter_name):
        self.parameter = parameter
        self.parameter_name = parameter_name

    def check(self):
        """Method raises ValueError exception if parameter is not equal to DataSet"""
        check_is_parameter_like_dataset(
            parameter=self.parameter, parameter_name=self.parameter_name
        )


class OptionalFilePathCheck(BaseInputArgumentChecker):
    """Class to check optional file_path-like parameters"""

    def __init__(self, parameter, parameter_name, expected_file_extension):
        self.parameter = parameter
        self.parameter_name = parameter_name
        self.expected_file_extensions = expected_file_extension

    def check(self):
        """Method raises ValueError exception if file path parameter is not equal to expected"""
        if self.parameter is not None:
            FilePathCheck(
                self.parameter, self.parameter_name, self.expected_file_extensions
            ).check()
