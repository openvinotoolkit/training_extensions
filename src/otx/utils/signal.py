# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Functions to append a signal handler."""

from __future__ import annotations

import os
import signal
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

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
