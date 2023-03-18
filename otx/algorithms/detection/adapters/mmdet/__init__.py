"""OTX Adapters - mmdet."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from .datasets.dataset import OTXDetDataset

# fmt: off
# isort: off
# FIXME: openvino pot library adds stream handlers to root logger
# which makes annoying duplicated logging
from mmdet.utils import get_root_logger  # pylint: disable=wrong-import-order
get_root_logger().propagate = False  # pylint: disable=wrong-import-order
# isort:on
# fmt: on

__all__ = ["OTXDetDataset"]
