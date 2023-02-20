"""OTX Adapters - mmseg."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .data import MPASegDataset
from .models import DetConB, DetConLoss, SelfSLMLP, SupConDetConB

# fmt: off
# isort: off
# FIXME: openvino pot library adds stream handlers to root logger
# which makes annoying duplicated logging
# pylint: disable=no-name-in-module,wrong-import-order
from mmseg.utils import get_root_logger  # type: ignore # (false positive)
get_root_logger().propagate = False
# fmt: off
# isort: on

__all__ = ["MPASegDataset", "DetConLoss", "SelfSLMLP", "DetConB", "SupConDetConB"]
