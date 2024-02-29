# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX action classification models."""

from .openvino_model import OTXOVActionCls
from .recognizers import OTXRecognizer3D

__all__ = ["OTXOVActionCls", "OTXRecognizer3D"]
