# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# This class and its supporting functions are adapted from the mmdet.
# Please refer to https://github.com/open-mmlab/mmdetection/

"""MMDet Box Decoders."""
from .delta_xywh_bbox_coder import DeltaXYWHBBoxCoder

__all__ = [
    "DeltaXYWHBBoxCoder",
]
