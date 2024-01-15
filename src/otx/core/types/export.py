# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""OTX export-related types definition."""

from __future__ import annotations

from enum import Enum


class OTXExportFormatType(str, Enum):
    """OTX export format type definition."""

    ONNX = "ONNX"
    OPENVINO = "OPENVINO"
    EXPORTABLE_CODE = "EXPORTABLE_CODE"


class OTXExportPrecisionType(str, Enum):
    """OTX export precision type definition."""

    FP16 = "FP16"
    FP32 = "FP32"
