"""OTX Adapters - mmseg."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .datasets import MPASegDataset
from .models import (
    ClassIncrEncoderDecoder,
    ConstantScalarScheduler,
    CrossEntropyLossWithIgnore,
    CustomFCNHead,
    DetConB,
    DetConLoss,
    LiteHRNet,
    MeanTeacherSegmentor,
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
    "CustomFCNHead",
    "DetConLoss",
    "SelfSLMLP",
    "ConstantScalarScheduler",
    "PolyScalarScheduler",
    "StepScalarScheduler",
    "DetConB",
    "CrossEntropyLossWithIgnore",
    "SupConDetConB",
    "ClassIncrEncoderDecoder",
    "MeanTeacherSegmentor",
]
