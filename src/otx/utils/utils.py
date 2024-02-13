# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OTX utility functions."""

import signal
import os
from decimal import Decimal
from dataclasses import dataclass
from typing import Callable, Any


@dataclass
class SigHandler:
    handler: Callable
    pid: int

_SIGNAL_HANDLERS: dict[signal.Signals, list] = {}


def append_signal_handler(sig_num: signal.Signals, sig_handler: Callable) -> None:
    """Append the handler for a signal. The function appended at last is called first.

    Args:
        sig_num (signal.Signals): Signal number to add a handler to.
        sig_handler (Callable): Callable function to be executed when the signal is sent.
    """
    _register_signal_handler(sig_num, sig_handler, -1)


def append_main_proc_signal_handler(sig_num: signal.Signals, sig_handler: Callable) -> None:
    """Append the handler for a signal triggered only by main process. The function appended at last is called first.

    It's almost same as append_signal_handler except that handler will be executed only by signal to
    process which registers handler.

    Args:
        sig_num (signal.Signals): Signal number to add a handler to.
        sig_handler (Callable): Callable function to be executed when the signal is sent.
    """
    _register_signal_handler(sig_num, sig_handler, os.getpid())


def _register_signal_handler(sig_num: signal.Signals, sig_handler: Callable, pid: int) -> None:
    if sig_num not in _SIGNAL_HANDLERS:
        old_sig_handler = signal.getsignal(sig_num)
        _SIGNAL_HANDLERS[sig_num] = [old_sig_handler]
        signal.signal(sig_num, _run_signal_handlers)

    _SIGNAL_HANDLERS[sig_num].insert(0, SigHandler(sig_handler, pid))


def _run_signal_handlers(sig_num: signal.Signals, frame) -> None:
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


def get_using_comma_seperated_key(key: str, target) -> Any:
    splited_key = key.split(".")
    for each_key in splited_key:
        target = target[each_key] if isinstance(target, dict) else getattr(target, each_key)
    return target


def set_using_comma_seperated_key(key: str, val: Any, target) -> None:
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
    return abs(Decimal(str(num)).as_tuple().exponent)
