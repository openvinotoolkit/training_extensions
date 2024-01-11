# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""OTX export type definition."""

from __future__ import annotations

from enum import Enum


class OTXExportFormat(str, Enum):
    """OTX export type definition."""

    OPENVINO = "OPENVINO"
    ONNX = "ONNX"
    EXPORTABLE_CODE = "EXPORTABLE_CODE"
