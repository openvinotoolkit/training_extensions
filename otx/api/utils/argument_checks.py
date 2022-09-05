"""Utils for checking functions and methods arguments."""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import inspect
import itertools
import typing
from abc import ABC, abstractmethod
from collections.abc import Sequence
from functools import wraps
from os.path import exists, splitext

import yaml
from numpy import floating
from omegaconf import DictConfig

IMAGE_FILE_EXTENSIONS = [
    ".bmp",
    ".dib",
    ".jpeg",
    ".jpg",
    ".jpe",
    ".jp2",
    ".png",
    ".webp",
    ".pbm",
    ".pgm",
    ".ppm",
    ".pxm",
    ".pnm",
    ".sr",
    ".ras",
    ".tiff",
    ".tif",
    ".exr",
    ".hdr",
    ".pic",
]


def get_bases(parameter) -> set:
    """Function to get set of all base classes of parameter."""

    def __get_bases(parameter_type):
        return [parameter_type.__name__] + list(
            itertools.chain.from_iterable(__get_bases(t1) for t1 in parameter_type.__bases__)
        )

    return set(__get_bases(type(parameter)))


def get_parameter_repr(parameter) -> str:
    """Function to get parameter representation."""
    try:
        parameter_str = repr(parameter)
    # pylint: disable=broad-except
    except Exception:
        parameter_str = "<unable to get parameter repr>"
    return parameter_str


def raise_value_error_if_parameter_has_unexpected_type(parameter, parameter_name, expected_type):
    """Function raises ValueError exception if parameter has unexpected type."""
    if isinstance(expected_type, typing.ForwardRef):
        expected_type = expected_type.__forward_arg__
    if isinstance(expected_type, str):
        parameter_types = get_bases(parameter)
        if not any(t == expected_type for t in parameter_types):
            parameter_str = get_parameter_repr(parameter)
            raise ValueError(
                f"Unexpected type of '{parameter_name}' parameter, expected: {expected_type}, "
                f"actual value: {parameter_str}"
            )
        return
    if expected_type == float:
        expected_type = (int, float, floating)
    if not isinstance(parameter, expected_type):
        parameter_type = type(parameter)
        parameter_str = get_parameter_repr(parameter)
        raise ValueError(
            f"Unexpected type of '{parameter_name}' parameter, expected: {expected_type}, actual: {parameter_type}, "
            f"actual value: {parameter_str}"
        )


def check_nested_elements_type(iterable, parameter_name, expected_type):
    """Function raises ValueError exception if one of elements in collection has unexpected type."""
    for element in iterable:
        check_parameter_type(
            parameter=element,
            parameter_name=f"nested {parameter_name}",
            expected_type=expected_type,
        )


def check_dictionary_keys_values_type(parameter, parameter_name, expected_key_class, expected_value_class):
    """Function raises ValueError exception if dictionary key or value has unexpected type."""
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


def check_nested_classes_parameters(parameter, parameter_name, origin_class, nested_elements_class):
    """Function to check type of parameters with nested elements."""
    # Checking origin class
    raise_value_error_if_parameter_has_unexpected_type(
        parameter=parameter, parameter_name=parameter_name, expected_type=origin_class
    )
    # Checking nested elements
    if origin_class == dict:
        if len(nested_elements_class) != 2:
            raise TypeError("length of nested expected types for dictionary should be equal to 2")
        key, value = nested_elements_class
        check_dictionary_keys_values_type(
            parameter=parameter,
            parameter_name=parameter_name,
            expected_key_class=key,
            expected_value_class=value,
        )
    if origin_class in [list, set, tuple, Sequence]:
        if origin_class == tuple:
            tuple_length = len(nested_elements_class)
            if tuple_length > 2:
                raise NotImplementedError("length of nested expected types for Tuple should not exceed 2")
            if tuple_length == 2:
                if nested_elements_class[1] != Ellipsis:
                    raise NotImplementedError("expected homogeneous tuple annotation")
                nested_elements_class = nested_elements_class[0]
        else:
            if len(nested_elements_class) != 1:
                raise TypeError("length of nested expected types for Sequence should be equal to 1")
        check_nested_elements_type(
            iterable=parameter,
            parameter_name=parameter_name,
            expected_type=nested_elements_class,
        )


def check_parameter_type(parameter, parameter_name, expected_type):
    """Function extracts nested expected types and raises ValueError exception if parameter has unexpected type."""
    # pylint: disable=W0212
    if expected_type in [typing.Any, inspect._empty]:  # type: ignore
        return
    if not isinstance(expected_type, typing._GenericAlias):  # type: ignore
        raise_value_error_if_parameter_has_unexpected_type(
            parameter=parameter,
            parameter_name=parameter_name,
            expected_type=expected_type,
        )
        return
    expected_type_dict = expected_type.__dict__
    origin_class = expected_type_dict.get("__origin__")
    nested_elements_class = expected_type_dict.get("__args__")
    # Union type with nested elements check
    if origin_class == typing.Union:
        expected_args = expected_type_dict.get("__args__")
        checks_counter = 0
        errors_counter = 0
        for expected_arg in expected_args:
            try:
                checks_counter += 1
                check_parameter_type(parameter, parameter_name, expected_arg)
            except ValueError:
                errors_counter += 1
        if errors_counter == checks_counter:
            actual_type = type(parameter)
            raise ValueError(
                f"Unexpected type of '{parameter_name}' parameter, expected: {expected_args}, "
                f"actual type: {actual_type}, actual value: {parameter}"
            )
    # Checking parameters with nested elements
    elif issubclass(origin_class, typing.Iterable):
        check_nested_classes_parameters(
            parameter=parameter,
            parameter_name=parameter_name,
            origin_class=origin_class,
            nested_elements_class=nested_elements_class,
        )


def check_input_parameters_type(custom_checks: typing.Optional[dict] = None):
    """Decorator to check input parameters type.

    Args:
        custom_checks: dictionary where key - name of parameter and
            value - custom check class
    """
    if custom_checks is None:
        custom_checks = {}

    def _check_input_parameters_type(function):
        @wraps(function)
        def validate(*args, **kwargs):
            # Forming expected types dictionary
            signature = inspect.signature(function)
            expected_types_map = signature.parameters
            if len(expected_types_map) < len(args):
                raise TypeError("Too many positional arguments")
            # Forming input parameters dictionary
            input_parameters_values_map = dict(zip(signature.parameters.keys(), args))
            for key, value in kwargs.items():
                if key in input_parameters_values_map:
                    raise TypeError(f"Duplication of the parameter {key} -- both in args and kwargs")
                input_parameters_values_map[key] = value
            # Checking input parameters type
            for parameter_name in expected_types_map:
                parameter = input_parameters_values_map.get(parameter_name)
                if parameter_name not in input_parameters_values_map:
                    default_value = expected_types_map.get(parameter_name).default
                    # pylint: disable=protected-access
                    if default_value != inspect._empty:  # type: ignore
                        parameter = default_value
                if parameter_name in custom_checks:
                    custom_check = custom_checks[parameter_name]
                    if custom_check is None:
                        continue
                    custom_check(parameter, parameter_name).check()
                else:
                    check_parameter_type(
                        parameter=parameter,
                        parameter_name=parameter_name,
                        expected_type=expected_types_map.get(parameter_name).annotation,
                    )
            return function(**input_parameters_values_map)

        return validate

    return _check_input_parameters_type


def check_file_extension(file_path: str, file_path_name: str, expected_extensions: list):
    """Function raises ValueError exception if file has unexpected extension."""
    file_extension = splitext(file_path)[1].lower()
    if file_extension not in expected_extensions:
        raise ValueError(
            f"Unexpected extension of {file_path_name} file. expected: {expected_extensions} actual: {file_extension}"
        )


def check_that_null_character_absents_in_string(parameter: str, parameter_name: str):
    r"""Function raises ValueError exception if null character: '\0' is specified in path to file."""
    if "\0" in parameter:
        raise ValueError(f"null char \\0 is specified in {parameter_name}: {parameter}")


def check_that_file_exists(file_path: str, file_path_name: str):
    """Function raises ValueError exception if file not exists."""
    if not exists(file_path):
        raise ValueError(f"File {file_path} specified in '{file_path_name}' parameter not exists")


def check_that_parameter_is_not_empty(parameter, parameter_name):
    """Function raises ValueError if parameter is empty."""
    if not parameter:
        raise ValueError(f"parameter {parameter_name} is empty")


def check_that_all_characters_printable(parameter, parameter_name, allow_crlf=False):
    """Function raises ValueError if one of string-parameter characters is not printable."""
    if not allow_crlf:
        all_characters_printable = all(c.isprintable() for c in parameter)
    else:
        all_characters_printable = all((c.isprintable() or c == "\n" or c == "\r") for c in parameter)
    if not all_characters_printable:
        raise ValueError(rf"parameter {parameter_name} has not printable symbols: {parameter}")


def check_is_parameter_like_dataset(parameter, parameter_name):
    """Checks if the parameter is like a dataset.

    Function raises ValueError exception if parameter does not have __len__, __getitem__ and get_subset attributes of
    DataSet-type object.
    """
    for expected_attribute in ("__len__", "__getitem__", "get_subset"):
        if not hasattr(parameter, expected_attribute):
            parameter_type = type(parameter)
            raise ValueError(
                f"parameter '{parameter_name}' is not like DatasetEntity, actual type: {parameter_type} which does "
                f"not have expected '{expected_attribute}' dataset attribute"
            )


def check_file_path(parameter, parameter_name, expected_file_extensions):
    """Function to check file path string objects."""
    raise_value_error_if_parameter_has_unexpected_type(
        parameter=parameter,
        parameter_name=parameter_name,
        expected_type=str,
    )
    check_that_parameter_is_not_empty(parameter=parameter, parameter_name=parameter_name)
    check_file_extension(
        file_path=parameter,
        file_path_name=parameter_name,
        expected_extensions=expected_file_extensions,
    )
    check_that_null_character_absents_in_string(parameter=parameter, parameter_name=parameter_name)
    check_that_all_characters_printable(parameter=parameter, parameter_name=parameter_name)
    check_that_file_exists(file_path=parameter, file_path_name=parameter_name)


def check_directory_path(parameter, parameter_name):
    """Function to check directory path string objects."""
    raise_value_error_if_parameter_has_unexpected_type(
        parameter=parameter,
        parameter_name=parameter_name,
        expected_type=str,
    )
    check_that_parameter_is_not_empty(parameter=parameter, parameter_name=parameter_name)
    check_that_null_character_absents_in_string(parameter=parameter, parameter_name=parameter_name)
    check_that_all_characters_printable(parameter=parameter, parameter_name=parameter_name)


class BaseInputArgumentChecker(ABC):
    """Abstract class to check input arguments."""

    @abstractmethod
    def check(self):
        """Abstract method to check input arguments."""
        raise NotImplementedError("The check is not implemented")


class InputConfigCheck(BaseInputArgumentChecker):
    """Class to check input config_parameters."""

    def __init__(self, parameter, parameter_name):
        self.parameter = parameter
        self.parameter_name = parameter_name

    def check(self):
        """Method raises ValueError exception if "input_config" parameter is not equal to expected."""
        raise_value_error_if_parameter_has_unexpected_type(
            parameter=self.parameter,
            parameter_name=self.parameter_name,
            expected_type=(str, DictConfig, dict),
        )
        check_that_parameter_is_not_empty(parameter=self.parameter, parameter_name=self.parameter_name)
        if isinstance(self.parameter, str):
            check_that_null_character_absents_in_string(parameter=self.parameter, parameter_name=self.parameter_name)
            # yaml-format string is specified
            if isinstance(yaml.safe_load(self.parameter), dict):
                check_that_all_characters_printable(
                    parameter=self.parameter,
                    parameter_name=self.parameter_name,
                    allow_crlf=True,
                )
            # Path to file is specified
            else:
                check_file_extension(
                    file_path=self.parameter,
                    file_path_name=self.parameter_name,
                    expected_extensions=[".yaml"],
                )
                check_that_all_characters_printable(parameter=self.parameter, parameter_name=self.parameter_name)
                check_that_file_exists(file_path=self.parameter, file_path_name=self.parameter_name)


class FilePathCheck(BaseInputArgumentChecker):
    """Class to check file_path-like parameters."""

    def __init__(self, parameter, parameter_name, expected_file_extension):
        self.parameter = parameter
        self.parameter_name = parameter_name
        self.expected_file_extensions = expected_file_extension

    def check(self):
        """Method raises ValueError exception if file path parameter is not equal to expected."""
        check_file_path(self.parameter, self.parameter_name, self.expected_file_extensions)


class OptionalFilePathCheck(BaseInputArgumentChecker):
    """Class to check optional file_path-like parameters."""

    def __init__(self, parameter, parameter_name, expected_file_extension):
        self.parameter = parameter
        self.parameter_name = parameter_name
        self.expected_file_extensions = expected_file_extension

    def check(self):
        """Method raises ValueError exception if file path parameter is not equal to expected."""
        if self.parameter is not None:
            check_file_path(self.parameter, self.parameter_name, self.expected_file_extensions)


class DatasetParamTypeCheck(BaseInputArgumentChecker):
    """Class to check DatasetEntity-type parameters."""

    def __init__(self, parameter, parameter_name):
        self.parameter = parameter
        self.parameter_name = parameter_name

    def check(self):
        """Method raises ValueError exception if parameter is not equal to Dataset."""
        check_is_parameter_like_dataset(parameter=self.parameter, parameter_name=self.parameter_name)


class OptionalImageFilePathCheck(OptionalFilePathCheck):
    """Class to check optional image file path parameters."""

    def __init__(self, parameter, parameter_name):
        super().__init__(
            parameter=parameter,
            parameter_name=parameter_name,
            expected_file_extension=IMAGE_FILE_EXTENSIONS,
        )


class YamlFilePathCheck(FilePathCheck):
    """Class to check optional yaml file path parameters."""

    def __init__(self, parameter, parameter_name):
        super().__init__(
            parameter=parameter,
            parameter_name=parameter_name,
            expected_file_extension=[".yaml"],
        )


class JsonFilePathCheck(FilePathCheck):
    """Class to check optional yaml file path parameters."""

    def __init__(self, parameter, parameter_name):
        super().__init__(
            parameter=parameter,
            parameter_name=parameter_name,
            expected_file_extension=[".json"],
        )


class DirectoryPathCheck(BaseInputArgumentChecker):
    """Class to check directory path parameters."""

    def __init__(self, parameter, parameter_name):
        self.parameter = parameter
        self.parameter_name = parameter_name

    def check(self):
        """Method raises ValueError exception if directory path parameter is not equal to expected."""
        check_directory_path(parameter=self.parameter, parameter_name=self.parameter_name)


class OptionalDirectoryPathCheck(BaseInputArgumentChecker):
    """Class to check optional directory path parameters."""

    def __init__(self, parameter, parameter_name):
        self.parameter = parameter
        self.parameter_name = parameter_name

    def check(self):
        """Method raises ValueError exception if directory path parameter is not equal to expected."""
        if self.parameter is not None:
            check_directory_path(parameter=self.parameter, parameter_name=self.parameter_name)
