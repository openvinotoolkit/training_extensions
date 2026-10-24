# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Backbone modules for getitune segmentation model."""

from .litehrnet import LiteHRNetBackbone
from .mscan import MSCAN

__all__ = ["MSCAN", "LiteHRNetBackbone"]
