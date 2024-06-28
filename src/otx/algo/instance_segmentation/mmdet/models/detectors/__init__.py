# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# This class and its supporting functions are adapted from the mmdet.
# Please refer to https://github.com/open-mmlab/mmdetection/

"""MMDet Detectors."""

from .mask_rcnn import MaskRCNN

__all__ = ["MaskRCNN"]
