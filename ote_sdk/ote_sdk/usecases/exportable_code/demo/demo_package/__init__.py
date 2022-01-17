"""
Initialization of demo package
"""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from .sync import SyncDemo
from .utils import create_model, create_output_converter

__all__ = ["SyncDemo", "create_model", "create_output_converter"]
