# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Utility files."""

from .signal import append_main_proc_signal_handler, append_signal_handler

__all__ = ["append_signal_handler", "append_main_proc_signal_handler"]
