# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from .config import SegmentationConfig
from .task import SegmentationInferenceTask, SegmentationTrainTask, SegmentationNNCFTask

# Load relevant extensions to registry
import mpa_tasks.extensions.datasets.mpa_seg_dataset

import mpa.seg

__all__ = [
    SegmentationConfig,
    SegmentationInferenceTask,
    SegmentationTrainTask,
    SegmentationNNCFTask,
    mpa_tasks.extensions.datasets.mpa_seg_dataset,
    mpa.seg
]
