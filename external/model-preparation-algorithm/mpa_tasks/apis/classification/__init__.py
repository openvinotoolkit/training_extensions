# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from .config import ClassificationConfig
from .task import ClassificationInferenceTask, ClassificationTrainTask

# Load relevant extensions to registry
import mpa_tasks.extensions.datasets.mpa_cls_dataset
import mpa_tasks.extensions.datasets.pipelines.mpa_cls_pipeline

import mpa.cls
