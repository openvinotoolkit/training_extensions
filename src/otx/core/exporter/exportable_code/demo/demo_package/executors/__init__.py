"""Initialization of executors."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from .asynchronous import AsyncExecutor
from .synchronous import SyncExecutor

__all__ = [
    "SyncExecutor",
    "AsyncExecutor",
]
