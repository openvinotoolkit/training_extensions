# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Collections of Utils for HPO."""

from __future__ import annotations

from typing import Literal


def left_vlaue_is_better(val1: int | float, val2: int | float, mode: Literal["max", "min"]) -> bool:
    """Check left value is better than right value.

    Whether check it's greather or lesser is changed depending on 'model'.

    Args:
        val1 : value to check that it's bigger than other value.
        val2 : value to check that it's bigger than other value.
        mode (Literal['max', 'min']): value to decide whether better means greater or lesser.

    Returns:
        bool: whether val1 is better than val2.
    """
    check_mode_input(mode)
    if mode == "max":
        return val1 > val2
    return val1 < val2


def check_positive(value: int | float, variable_name: str | None = None, error_message: str | None = None) -> None:
    """Validate that value is positivle.

    Args:
        value (int | float): value to validate.
        variable_name (str | None, optional): name of value. It's used for error message. Defaults to None.
        error_message (str | None, optional): Error message to use when type is different. Defaults to None.

    Raises:
        ValueError: If value isn't positive, the error is raised.
    """
    if value <= 0:
        if error_message is not None:
            message = error_message
        elif variable_name:
            message = f"{variable_name} should be positive.\nyour value : {value}"
        else:
            raise ValueError
        raise ValueError(message)


def check_not_negative(value: int | float, variable_name: str | None = None, error_message: str | None = None) -> None:
    """Validate that value isn't negative.

    Args:
        value (int | float): value to validate.
        variable_name (str | None, optional): name of value. It's used for error message. Defaults to None.
        error_message (str | None, optional): Error message to use when type is different. Defaults to None.

    Raises:
        ValueError: If value is negative, the error is raised.
    """
    if value < 0:
        if error_message is not None:
            message = error_message
        elif variable_name:
            message = f"{variable_name} should be positive.\nyour value : {value}"
        else:
            raise ValueError
        raise ValueError(message)


def check_mode_input(mode: str) -> None:
    """Validate that mode is 'max' or 'min'.

    Args:
        mode (str): string to validate.

    Raises:
        ValueError: If 'mode' is not both 'max' and 'min', the error is raised.
    """
    if mode not in ["max", "min"]:
        error_msg = f"mode should be max or min.\nYour value : {mode}"
        raise ValueError(error_msg)
