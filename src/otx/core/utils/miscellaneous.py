# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""A collection of miscellaneous utility functions."""

from typing import Callable, TypeVar

_T = TypeVar("_T")
_V = TypeVar("_V")


def ensure_callable(func: Callable[[_T], _V]) -> Callable[[_T], _V]:
    """If the given input is not callable, raise TypeError."""
    if not callable(func):
        raise TypeError(func)
    return func
