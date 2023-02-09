"""HPO utility."""

# Copyright (C) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from typing import Any, List, Optional, Tuple, Union


def check_type(
    value: type,
    available_type: Union[List, Tuple],
    variable_name: Optional[str] = None,
    error_message: Optional[str] = None,
):
    """Validate value type. If type is different, raise an error.

    Args:
        value (type): value to type check.
        available_type (Union[List, Tuple]): types expected that the value has.
        variable_name (Optional[str], optional): name of value. It's used for error message. Defaults to None.
        error_message (Optional[str], optional): Error message to use when type is different. Defaults to None.

    Raises:
        TypeError: If type of value is different with the available type, then error is raised.
    """
    if not isinstance(value, available_type):  # type: ignore
        if not isinstance(available_type, tuple):
            available_type = [available_type]
        if error_message is not None:
            message = error_message
        elif variable_name is not None:
            message = (
                f"{variable_name} should be "
                + " or ".join([str(val) for val in available_type])
                + f". Current {variable_name} type is {type(value)}"
            )
        else:
            raise TypeError
        raise TypeError(message)


def check_positive(value: Any, variable_name: Optional[str] = None, error_message: Optional[str] = None):
    """Validate that value is positivle.

    Args:
        value (Any): value to validate.
        variable_name (Optional[str], optional): name of value. It's used for error message. Defaults to None.
        error_message (Optional[str], optional): Error message to use when type is different. Defaults to None.

    Raises:
        ValueError: If value isn't positive, the error is raised.
    """
    if value <= 0:
        if error_message is not None:
            message = error_message
        elif variable_name:
            message = f"{variable_name} should be positive.\n" f"your value : {value}"
        else:
            raise ValueError
        raise ValueError(message)


def check_not_negative(value: Any, variable_name: Optional[str] = None, error_message: Optional[str] = None):
    """Validate that value isn't negative.

    Args:
        value (Any): value to validate.
        variable_name (Optional[str], optional): name of value. It's used for error message. Defaults to None.
        error_message (Optional[str], optional): Error message to use when type is different. Defaults to None.

    Raises:
        ValueError: If value is negative, the error is raised.
    """
    if value < 0:
        if error_message is not None:
            message = error_message
        elif variable_name:
            message = f"{variable_name} should be positive.\n" f"your value : {value}"
        else:
            raise ValueError
        raise ValueError(message)


def check_mode_input(mode: str):
    """Validate that mode is 'max' or 'min'.

    Args:
        mode (str): string to validate.

    Raises:
        ValueError: If 'mode' is not both 'max' and 'min', the error is raised.
    """
    if mode not in ["max", "min"]:
        raise ValueError("mode should be max on min.\n" f"Your value : {mode}")


def left_is_better(val1, val2, mode: str) -> bool:
    """Check left value is better than right value.

    Whether check it's greather or lesser is changed depending on 'model'.

    Args:
        val1 : value to check that it's bigger than other value.
        val2 : value to check that it's bigger than other value.
        mode (str): value to decide whether better means greater or lesser.

    Returns:
        bool: whether val1 is better than val2.
    """
    check_mode_input(mode)
    if mode == "max":
        return val1 > val2
    else:
        return val1 < val2
