"""OTX Adapters - mmseg."""


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

from .datasets import MPASegDataset
from .models import (
    ConstantScalarScheduler,
    CrossEntropyLossWithIgnore,
    DetConB,
    DetConLoss,
    LiteHRNet,
    MeanTeacherSegmentor,
    MMOVBackbone,
    MMOVDecodeHead,
    PolyScalarScheduler,
    SelfSLMLP,
    StepScalarScheduler,
    SupConDetConB,
)

# fmt: off
# isort: off
# FIXME: openvino pot library adds stream handlers to root logger
# which makes annoying duplicated logging
# pylint: disable=no-name-in-module,wrong-import-order
from mmseg.utils import get_root_logger  # type: ignore # (false positive)
get_root_logger().propagate = False
# fmt: off
# isort: on

__all__ = [
    "MPASegDataset",
    "LiteHRNet",
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
]
