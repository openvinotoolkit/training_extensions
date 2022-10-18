# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import mpa.cls as MPAClassification

from .openvino import ClassificationOpenVINOTask
from .inference import ClassificationInferenceTask
from .train import ClassificationTrainTask

__all__ = [
    "MPAClassification",
    "ClassificationOpenVINOTask",
    "ClassificationInferenceTask",
    "ClassificationTrainTask",
]
