# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from .config import DetectionConfig
from .task import DetectionInferenceTask, DetectionTrainTask

# Load relevant extensions to registry
import mpa_tasks.extensions.datasets.mpa_det_dataset
