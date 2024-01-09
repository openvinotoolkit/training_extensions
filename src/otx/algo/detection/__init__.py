# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Custom model implementations for detection task."""

from . import backbones, heads
from .otx_ssd import OTXSSD

__all__ = ["backbones", "heads", "OTXSSD"]
