"""OTX Adapters - openvino."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .task import (
    OpenVINOSegmentationInferencer,
    OpenVINOSegmentationTask,
)

__all__ = [
    "OpenVINOSegmentationTask",
    "OpenVINOSegmentationInferencer",
]
