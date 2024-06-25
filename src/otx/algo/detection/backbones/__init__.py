# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Custom backbone implementations for detection task."""

from .csp_darknet import CSPDarknet
from .cspnext import CSPNeXt
from .pytorchcv_backbones import build_model_including_pytorchcv

__all__ = ["CSPDarknet", "CSPNeXt", "build_model_including_pytorchcv"]
