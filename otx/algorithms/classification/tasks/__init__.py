# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import mpa.cls as MPAClassification

from .datasets import MPAClsDataset
from .openvino_task import ClassificationOpenVINOTask
from .task import (
    ClassificationInferenceTask,
    ClassificationNNCFTask,
    ClassificationTrainTask,
)

__all__ = [
    "MPAClassification",
    "MPAClsDataset",
    "ClassificationOpenVINOTask",
    "ClassificationInferenceTask",
    "ClassificationTrainTask",
    "ClassificationNNCFTask",
]
