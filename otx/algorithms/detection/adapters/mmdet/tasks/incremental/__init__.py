"""Initialize incremental learning for OTX Detection."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from .inferrer import IncrDetectionInferrer
from .trainer import IncrDetectionTrainer

__all__ = ["IncrDetectionTrainer", "IncrDetectionInferrer"]
