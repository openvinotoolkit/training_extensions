# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import mpa.det

# Load relevant extensions to registry
import mpa_tasks.extensions.datasets.mpa_det_dataset

from .config import DetectionConfig
from .task import DetectionInferenceTask, DetectionNNCFTask, DetectionTrainTask

__all__ = [
    DetectionConfig,
    DetectionInferenceTask,
    DetectionTrainTask,
    DetectionNNCFTask,
    mpa_tasks.extensions.datasets.mpa_det_dataset,
    mpa.det,
]
