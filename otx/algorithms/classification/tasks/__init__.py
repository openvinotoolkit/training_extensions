# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import mpa.cls as MPAClassification

from .datasets import MPAClsDataset
from .task import (
    ClassificationInferenceTask,
    ClassificationNNCFTask,
    ClassificationTrainTask,
)

__all__ = [
    "MPAClassification",
    "MPAClsDataset",
    "ClassificationInferenceTask",
    "ClassificationTrainTask",
    "ClassificationNNCFTask",
]
