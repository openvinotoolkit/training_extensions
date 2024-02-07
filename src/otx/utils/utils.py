# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OTX utility functions."""

import signal
from functools import partial
from typing import Callable, Any


def append_signal_handler(sig_num: signal.Signals, sig_handler: Callable) -> None:
    """Append the handler for a signal. The function appended at last is called first.

    Args:
        sig_num (signal.Signals): Signal number to add a handler to.
        sig_handler (Callable): Callable function to be executed when the signal is sent.
    """
    old_sig_handler = signal.getsignal(sig_num)

    def helper(*args, old_func: Callable, **kwargs) -> None:
        sig_handler(*args, **kwargs)
        if old_func == signal.SIG_DFL:
            signal.signal(sig_num, signal.SIG_DFL)
            signal.raise_signal(sig_num)
        elif callable(old_func):
            old_func(*args, **kwargs)

    signal.signal(sig_num, partial(helper, old_func=old_sig_handler))


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
