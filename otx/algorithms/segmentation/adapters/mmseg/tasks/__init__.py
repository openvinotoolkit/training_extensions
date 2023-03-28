"""Initialize OTX Segmentation with MMSEG."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import otx.algorithms.common.adapters.mmcv.hooks
import otx.algorithms.segmentation.adapters.mmseg
import otx.algorithms.segmentation.adapters.mmseg.models
import otx.algorithms.segmentation.adapters.mmseg.models.schedulers
from otx.algorithms.segmentation.adapters.mmseg.tasks.incremental import (
    IncrSegInferrer,
    IncrSegTrainer,
)
from otx.algorithms.segmentation.adapters.mmseg.tasks.semisl import (
    SemiSLSegExporter,
    SemiSLSegInferrer,
    SemiSLSegTrainer,
)

# flake8: noqa
from . import exporter, inferrer, stage, trainer
