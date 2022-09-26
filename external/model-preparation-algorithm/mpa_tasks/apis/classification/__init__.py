# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import mpa.cls

# Load relevant extensions to registry
import mpa_tasks.extensions.datasets.mpa_cls_dataset
import mpa_tasks.extensions.datasets.pipelines.mpa_cls_pipeline

from .config import ClassificationConfig
from .task import (
    ClassificationInferenceTask,
    ClassificationNNCFTask,
    ClassificationTrainTask,
)

__all__ = [
    ClassificationConfig,
    ClassificationInferenceTask,
    ClassificationTrainTask,
    ClassificationNNCFTask,
    mpa_tasks.extensions.datasets.mpa_cls_dataset,
    mpa_tasks.extensions.datasets.pipelines.mpa_cls_pipeline,
    mpa.cls,
]
