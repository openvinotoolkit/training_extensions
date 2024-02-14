# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OTX utility functions."""

from __future__ import annotations

import os
import signal
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from types import FrameType


@dataclass
class SigHandler:
    """Signal handler dataclass having handler function and pid which registers the handler."""

    handler: Callable
    pid: int


_SIGNAL_HANDLERS: dict[int, list] = {}


def append_signal_handler(sig_num: int, sig_handler: Callable) -> None:
    """Append the handler for a signal. The function appended at last is called first.

    Args:
        sig_num (signal.Signals): Signal number to add a handler to.
        sig_handler (Callable): Callable function to be executed when the signal is sent.
    """
    _register_signal_handler(sig_num, sig_handler, -1)


def append_main_proc_signal_handler(sig_num: int, sig_handler: Callable) -> None:
    """Append the handler for a signal triggered only by main process. The function appended at last is called first.

    It's almost same as append_signal_handler except that handler will be executed only by signal to
    process which registers handler.

    Args:
        sig_num (signal.Signals): Signal number to add a handler to.
        sig_handler (Callable): Callable function to be executed when the signal is sent.
    """
    _register_signal_handler(sig_num, sig_handler, os.getpid())


def _register_signal_handler(sig_num: int, sig_handler: Callable, pid: int) -> None:
    if sig_num not in _SIGNAL_HANDLERS:
        old_sig_handler = signal.getsignal(sig_num)
        _SIGNAL_HANDLERS[sig_num] = [old_sig_handler]
        signal.signal(sig_num, _run_signal_handlers)

    _SIGNAL_HANDLERS[sig_num].insert(0, SigHandler(sig_handler, pid))


def _run_signal_handlers(sig_num: int, frame: FrameType | None) -> None:
    pid = os.getpid()
    for handler in _SIGNAL_HANDLERS[sig_num]:
        if handler == signal.SIG_DFL:
            signal.signal(sig_num, signal.SIG_DFL)
            signal.raise_signal(sig_num)
        elif isinstance(handler, SigHandler):
            if handler.pid < 0 or handler.pid == pid:
                handler.handler(sig_num, frame)
        else:
            handler(sig_num, frame)


def get_using_dot_delimited_key(key: str, target: Any) -> Any:  # noqa: ANN401
    """Get values of attribute in target object using dot delimited key.

    For example, if key is "a.b.c", then get a value of 'target.a.b.c'.
    target should be object having attributes or dictionary.

    Args:
        key (str): dot delimited key.
        val (Any): value to set.
        target (Any): target to set value to.
    """
    splited_key = key.split(".")
    for each_key in splited_key:
        target = target[each_key] if isinstance(target, dict) else getattr(target, each_key)
    return target


def set_using_dot_delimited_key(key: str, val: Any, target: Any) -> None:  # noqa: ANN401
    """Set values to attribute in target object using dot delimited key.

    For example, if key is "a.b.c", then value is set at 'target.a.b.c'.
    target should be object having attributes or dictionary.

    Args:
        key (str): dot delimited key.
        val (Any): value to set.
        target (Any): target to set value to.
    """
    splited_key = key.split(".")
    for each_key in splited_key[:-1]:
        target = target[each_key] if isinstance(target, dict) else getattr(target, each_key)

    if isinstance(target, dict):
        target[splited_key[-1]] = val
    else:
        setattr(target, splited_key[-1], val)


def get_decimal_point(num: float) -> int:
    """Find a decimal point from the given float.

    Args:
        num (float): float to find a decimal point from.

    Returns:
        int: decimal point.
    """
    if isinstance((exponent := Decimal(str(num)).as_tuple().exponent), int):
        return abs(exponent)
    error_msg = f"Can't get an exponent from {num}."
    raise ValueError(error_msg)
