# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from .task import (
    ClassificationInferenceTask,
    ClassificationNNCFTask,
    ClassificationTrainTask,
)

__all__ = [
    ClassificationInferenceTask,
    ClassificationTrainTask,
    ClassificationNNCFTask,
]
