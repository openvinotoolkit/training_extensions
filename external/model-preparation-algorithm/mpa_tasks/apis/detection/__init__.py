# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from .config import DetectionConfig
from .task import DetectionInferenceTask, DetectionTrainTask, DetectionNNCFTask

# Load relevant extensions to registry
import mpa_tasks.extensions.datasets.mpa_det_dataset

import mpa.det

__all__ = [
    DetectionConfig,
    DetectionInferenceTask,
    DetectionTrainTask,
    DetectionNNCFTask,
    mpa_tasks.extensions.datasets.mpa_det_dataset,
    mpa.det,
]
