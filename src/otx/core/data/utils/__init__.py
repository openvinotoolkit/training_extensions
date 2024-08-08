# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Utility modules for core data modules."""

from .utils import adapt_input_size_to_dataset, adapt_tile_config

__all__ = ["adapt_tile_config", "adapt_input_size_to_dataset"]
