"""OTX Adapters - mmdet."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .data import MPADetDataset

# FIXME: openvino pot library adds stream handlers to root logger
# which makes annoying duplicated logging
from mmdet.utils import get_root_logger
get_root_logger().propagate = False

__all__ = ["MPADetDataset"]
