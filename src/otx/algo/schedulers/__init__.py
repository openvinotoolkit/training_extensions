# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Custom schedulers for the OTX2.0."""

from .warmup_schedulers import WarmupReduceLROnPlateau

__all__ = ["WarmupReduceLROnPlateau"]
