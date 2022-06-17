# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from .config import SegmentationConfig
from .task import SegmentationInferenceTask, SegmentationTrainTask

# Load relevant extensions to registry
import mpa_tasks.extensions.datasets.mpa_seg_dataset

import mpa.seg
