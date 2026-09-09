# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Utility functions."""

from __future__ import annotations

from typing import Any, Callable, TypeVar

_T = TypeVar("_T")
_V = TypeVar("_V")


def is_ckpt_for_finetuning(ckpt: dict) -> bool:
    """Check the checkpoint will be used to finetune.

    Args:
        ckpt (dict): the checkpoint file

    Returns:
        bool: True means the checkpoint will be used to finetune.
    """
    return "state_dict" in ckpt


def remove_state_dict_prefix(state_dict: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Remove prefix from state_dict keys."""
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace(prefix, "")
        new_state_dict[new_key] = value
    return new_state_dict


def ensure_callable(func: Callable[[_T], _V]) -> Callable[[_T], _V]:
    """If the given input is not callable, raise TypeError."""
    if not callable(func):
        raise TypeError(func)
    return func
