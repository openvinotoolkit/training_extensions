from typing import Any, Optional, Union, Tuple


def check_type(
    value: Any,
    available_type: Union[type, Tuple],
    variable_name: Optional[str] = None,
    error_message: Optional[str] = None
):
    """Validate value type and if not, raise error."""
    if not isinstance(value, available_type):
        if not isinstance(available_type, tuple):
            available_type = [available_type]
        if error_message is not None:
            message = error_message
        elif variable_name is not None:
            message = (
                f"{variable_name} should be " +
                " or ".join([str(val) for val in available_type]) +
                f". Current {variable_name} type is {type(value)}"
            )
        else:
            raise TypeError
        raise TypeError(message)

def check_positive(
    value: Any,
    variable_name: Optional[str] = None,
    error_message: Optional[str] = None
):
    if value <= 0:
        if error_message is not None:
            message = error_message
        elif variable_name:
            message = (
                f"{variable_name} should be positive.\n"
                f"your value : {value}"
            )
        else:
            raise ValueError
        raise ValueError(message)

def check_not_negative(
    value: Any,
    variable_name: Optional[str] = None,
    error_message: Optional[str] = None
):
    if value < 0:
        if error_message is not None:
            message = error_message
        elif variable_name:
            message = (
                f"{variable_name} should be positive.\n"
                f"your value : {value}"
            )
        else:
            raise ValueError
        raise ValueError(message)

def check_mode_input(mode):
    if mode not in ["max", "min"]:
        raise ValueError(
            "mode should be max on min.\n"
            f"Your value : {mode}"
        )

def left_is_better(val1, val2, mode):
    check_mode_input(mode)
    if mode == "max":
        return val1 > val2
    else:
        return val1 < val2

def dummy_obj(**kwargs):
    return 0