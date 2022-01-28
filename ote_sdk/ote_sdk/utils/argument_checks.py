"""
Utils for checking functions and methods arguments
"""

from os.path import exists

from omegaconf import DictConfig
from yaml import safe_load


def check_parameter_type(parameter, parameter_name, expected_type):
    """Function raises ValueError exception if parameter has unexpected type"""
    if not isinstance(parameter, expected_type):
        parameter_type = type(parameter)
        raise ValueError(
            f"Unexpected type of '{parameter_name}' parameter, expected: {expected_type}, actual: {parameter_type}"
        )


def check_required_parameters_type(parameter_name_expected_type: list):
    """
    Function raises ValueError exception if required parameters have unexpected type
    :param parameter_name_expected_type: list with tuples that contain parameter, name for exception message and
    expected type
    """
    for parameter, name, expected_type in parameter_name_expected_type:
        check_parameter_type(
            parameter=parameter, parameter_name=name, expected_type=expected_type
        )


def check_optional_parameters_type(parameter_name_expected_type: list):
    """
    Function checks if optional parameters exist and raises ValueError exception if one of them has unexpected type
    :param parameter_name_expected_type: list with tuples that contain optional parameter, name for exception message
    and expected type
    """
    for parameter, name, expected_type in parameter_name_expected_type:
        if parameter:
            check_parameter_type(
                parameter=parameter, parameter_name=name, expected_type=expected_type
            )


def check_required_and_optional_parameters_type(
    required_parameters: list, optional_parameters: list
):
    """Function raises ValueError exception if required or optional parameter has unexpected type"""
    check_required_parameters_type(required_parameters)
    check_optional_parameters_type(optional_parameters)


def check_nested_elements_type(iterable, parameter_name, expected_type):
    """Function raises ValueError exception if one of elements in collection has unexpected type"""
    for element in iterable:
        check_parameter_type(
            parameter=element,
            parameter_name=f"nested {parameter_name}",
            expected_type=expected_type,
        )


def check_several_lists_elements_type(parameter_name_expected_type: list):
    """
    Function checks if parameters lists exist and raises ValueError exception if lists elements have unexpected type
    :param parameter_name_expected_type: list with tuples that contain parameter with nested elements, name for
    exception message and expected type
    """
    for parameter, name, expected_type in parameter_name_expected_type:
        if parameter:
            check_nested_elements_type(
                iterable=parameter, parameter_name=name, expected_type=expected_type
            )


def check_parameter_str_class_name(parameter, parameter_name, expected_class_names):
    """Function raises ValueError exception if string class name is not equal to expected"""
    parameter_class_name = type(parameter).__name__
    if (parameter_class_name in expected_class_names) is None:
        raise ValueError(
            f"Unexpected type of '{parameter_name}' parameter, expected: {expected_class_names}, actual: "
            f"{parameter_class_name}"
        )


def check_dictionary_keys_values_type(
    parameter, parameter_name, expected_key_class, expected_value_class
):
    """Function raises ValueError exception if dictionary keys or values have unexpected type"""
    for key, value in parameter.items():
        parameter_type = type(key)
        if not isinstance(key, expected_key_class):
            raise ValueError(
                f"Unexpected type of nested '{parameter_name}' dictionary key, expected: {expected_key_class}, "
                f"actual: {parameter_type}"
            )
        parameter_type = type(value)
        if not isinstance(value, expected_value_class):
            raise ValueError(
                f"Unexpected type of nested '{parameter_name}' dictionary value, expected: {expected_value_class}, "
                f"actual: {parameter_type}"
            )


def check_several_dictionaries_keys_values_type(parameter_name_expected_type: list):
    """
    Function checks if parameters dictionaries exist and raises ValueError exception if their key or value have
    unexpected type
    :param parameter_name_expected_type: list with tuples that contain dictionary parameter, name for exception message
    and expected type
    """
    for (
        parameter,
        name,
        expected_key_class,
        expected_value_class,
    ) in parameter_name_expected_type:
        if parameter:
            check_dictionary_keys_values_type(
                parameter=parameter,
                parameter_name=name,
                expected_key_class=expected_key_class,
                expected_value_class=expected_value_class,
            )


def check_that_string_not_empty(string: str, parameter_name: str):
    """Function raises ValueError exception if string parameter is empty"""
    if string == "":
        raise ValueError(f"Empty string is specified as {parameter_name} parameter")


def check_file_extension(
    file_path: str, file_path_name: str, expected_extensions: list
):
    """Function raises ValueError exception if file has unexpected extension"""
    file_extension = file_path.split(".")[-1].lower()
    if file_extension not in expected_extensions:
        raise ValueError(
            f"Unexpected extension of {file_path_name} file. expected: {expected_extensions} actual: {file_extension}"
        )


def check_that_null_character_absents_in_path(file_path: str, file_path_name: str):
    """Function raises ValueError exception if null character: '\0' is specified in path to file"""
    if "\\0" in file_path:
        raise ValueError(f"\\0 is specified in {file_path_name}: {file_path}")


def check_that_file_exists(file_path: str, file_path_name: str):
    """Function raises ValueError exception if file not exists"""
    if not exists(file_path):
        raise ValueError(
            f"File {file_path} specified in '{file_path_name}' parameter not exists"
        )


def check_file_path(file_path: str, file_path_name: str, expected_extensions: list):
    """
    Function raises ValueError exception if non-string object is specified as file path, if file has unexpected
    extension or if file not exists
    """
    check_parameter_type(
        parameter=file_path, parameter_name=file_path_name, expected_type=str
    )
    check_that_string_not_empty(string=file_path, parameter_name=file_path_name)
    check_file_extension(
        file_path=file_path,
        file_path_name=file_path_name,
        expected_extensions=expected_extensions,
    )
    check_that_null_character_absents_in_path(
        file_path=file_path, file_path_name=file_path_name
    )
    check_that_file_exists(file_path=file_path, file_path_name=file_path_name)


def check_input_config_parameter(input_config):
    """
    Function raises ValueError exception if "input_config" parameter is not equal to expected
    """
    parameter_name = "input_config"
    check_parameter_type(
        parameter=input_config,
        parameter_name=parameter_name,
        expected_type=(str, DictConfig, dict),
    )
    if isinstance(input_config, str):
        check_that_string_not_empty(string=input_config, parameter_name=parameter_name)
        if isinstance(safe_load(input_config), str):
            check_file_extension(
                file_path=input_config,
                file_path_name=parameter_name,
                expected_extensions=["yaml"],
            )
            check_that_null_character_absents_in_path(
                file_path=input_config, file_path_name=parameter_name
            )
            check_that_file_exists(
                file_path=input_config, file_path_name=parameter_name
            )
    if isinstance(input_config, dict):
        if input_config == {}:
            raise ValueError(
                "Empty dictionary is specified as 'input_config' parameter"
            )
