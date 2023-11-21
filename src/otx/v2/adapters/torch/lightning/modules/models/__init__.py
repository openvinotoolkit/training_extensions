"""Lightning model."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from otx.v2.adapters.torch.lightning.registry import MODELS

from . import anomaly
from .visual_prompters import SegmentAnything

__all__ = ["anomaly", "SegmentAnything", "MODELS"]
