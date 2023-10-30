"""Adapters for OTX Common Algorithm. - mmseg.model."""

# Copyright (C) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from .backbones import MSCAN, LiteHRNet, MMOVBackbone
from .heads import DetConHead, LightHamHead, MMOVDecodeHead
from .losses import CrossEntropyLossWithIgnore, DetConLoss
from .necks import SelfSLMLP
from .schedulers import (
    ConstantScalarScheduler,
    PolyScalarScheduler,
    StepScalarScheduler,
)
from .segmentors import (
    DetConB,
    MeanTeacherSegmentor,
    SupConDetConB,
)

__all__ = [
    "LiteHRNet",
    "MSCAN",
    "MMOVBackbone",
    "MMOVDecodeHead",
    "DetConLoss",
    "SelfSLMLP",
    "ConstantScalarScheduler",
    "PolyScalarScheduler",
    "StepScalarScheduler",
    "DetConB",
    "CrossEntropyLossWithIgnore",
    "SupConDetConB",
    "MeanTeacherSegmentor",
    "DetConHead",
    "LightHamHead",
]
