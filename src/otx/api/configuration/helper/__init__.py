"""This module contains the configuration helper functions.

These can be used to create, convert or interact with OTX configuration objects or dictionaries, yaml strings or yaml
files representing those objects.
"""
# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


from .convert import convert
from .create import create
from .substitute import substitute_values, substitute_values_for_lifecycle
from .utils import config_to_bytes, flatten_config_values, merge_a_into_b
from .validate import validate

__all__ = [
    "create",
    "config_to_bytes",
    "validate",
    "convert",
    "substitute_values",
    "substitute_values_for_lifecycle",
    "flatten_config_values",
    "merge_a_into_b",
]
