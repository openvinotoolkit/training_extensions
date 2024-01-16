# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""X3DFastRCNN model implementation."""
from __future__ import annotations

from otx.algo.utils.mmconfig import read_mmconfig
from otx.core.model.entity.action_detection import MMActionCompatibleModel


class X3DFastRCNN(MMActionCompatibleModel):
    """X3D Model."""

    def __init__(self, num_classes: int, topk: int | tuple[int]):
        config = read_mmconfig("x3d_fastrcnn")
        config.roi_head.bbox_head.topk = topk
        super().__init__(num_classes=num_classes, config=config)
