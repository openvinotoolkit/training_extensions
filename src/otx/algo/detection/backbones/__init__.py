# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Custom backbone implementations for detection task."""

from . import pytorchcv_backbones
from .csp_darknet import CSPDarknet

__all__ = ["pytorchcv_backbones", "CSPDarknet"]
