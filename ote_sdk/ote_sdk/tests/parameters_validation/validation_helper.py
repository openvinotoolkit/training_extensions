"""
Common functions for input parameters validation tests
"""

import pytest


def check_value_error_exception_raised(
    correct_parameters: dict, unexpected_values: list, class_or_function
) -> None:
    """
    Function checks that ValueError exception is raised when unexpected type values are specified as parameters for
    methods or functions
    """
    for key, value in unexpected_values:
        incorrect_parameters_dict = dict(correct_parameters)
        incorrect_parameters_dict[key] = value
        with pytest.raises(ValueError):
            class_or_function(**incorrect_parameters_dict)
