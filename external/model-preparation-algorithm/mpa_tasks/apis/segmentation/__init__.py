# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import mpa.seg

# Load relevant extensions to registry
import mpa_tasks.extensions.datasets.mpa_seg_dataset

from .config import SegmentationConfig
from .task import SegmentationInferenceTask, SegmentationNNCFTask, SegmentationTrainTask

__all__ = [
    SegmentationConfig,
    SegmentationInferenceTask,
    SegmentationTrainTask,
    SegmentationNNCFTask,
    mpa_tasks.extensions.datasets.mpa_seg_dataset,
    mpa.seg,
]
