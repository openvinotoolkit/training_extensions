# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX custom algorithms, e.g., model, losses, hook, etc..."""

from . import classification, detection, segmentation

__all__ = ["classification", "detection", "segmentation"]
