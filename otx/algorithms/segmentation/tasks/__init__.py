# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import mpa.seg as MPASegmentation

from .openvino_task import OpenVINOSegmentationTask
from .task import (
    SegmentationInferenceTask,
    SegmentationTrainTask,
    SegmentationNNCFTask,
)

__all__ = [
    "MPASegmentation",
    "SegmentationInferenceTask",
    "SegmentationTrainTask",
    "SegmentationNNCFTask",
    "OpenVINOSegmentationTask"
]
