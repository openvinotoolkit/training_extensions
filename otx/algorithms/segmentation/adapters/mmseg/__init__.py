"""OTX Adapters - mmseg."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .data import MPASegDataset
from .models import DetConB, DetConLoss, SelfSLMLP

# FIXME: openvino pot library adds stream handlers to root logger
# which makes annoying duplicated logging
from mmseg.utils import get_root_logger
get_root_logger().propagate = False

__all__ = ["MPASegDataset", "DetConLoss", "SelfSLMLP", "DetConB"]
