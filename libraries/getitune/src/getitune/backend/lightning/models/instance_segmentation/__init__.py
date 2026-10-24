# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Module for getitune instance segmentation models."""

from .edgecrafter_inst import EdgeCrafterInst
from .maskrcnn import MaskRCNN
from .maskrcnn_tv import MaskRCNNTV
from .rfdetr_inst import RFDETRInst
from .rtmdet_inst import RTMDetInst

__all__ = ["EdgeCrafterInst", "MaskRCNN", "MaskRCNNTV", "RFDETRInst", "RTMDetInst"]
