# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""LiteHRNet model implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from otx.algo.segmentation.backbones import LiteHRNet
from otx.algo.segmentation.heads import CustomFCNHead
from otx.algo.utils.support_otx_v1 import OTXv1Helper
from otx.core.model.segmentation import TorchVisionCompatibleModel

from .base_model import BaseSegmNNModel

if TYPE_CHECKING:
    from torch import nn


class LiteHRNetS(BaseSegmNNModel):
    """LiteHRNetS Model."""

    @property
    def ignore_scope(self) -> dict[str, str | dict[str, list[str]]]:
        """The ignored scope for LiteHRNetS."""
        ignored_scope_names = [
            "/model/backbone/stage0/stage0.0/layers/layers.0/cross_resolution_weighting/Mul",
            "/model/backbone/stage0/stage0.0/layers/layers.0/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage0/stage0.0/layers/layers.1/cross_resolution_weighting/Mul",
            "/model/backbone/stage0/stage0.0/layers/layers.1/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage0/stage0.0/Add_1",
            "/model/backbone/stage0/stage0.1/layers/layers.0/cross_resolution_weighting/Mul",
            "/model/backbone/stage0/stage0.1/layers/layers.0/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage0/stage0.1/layers/layers.1/cross_resolution_weighting/Mul",
            "/model/backbone/stage0/stage0.1/layers/layers.1/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage0/stage0.1/Add_1",
            "/model/backbone/stage0/stage0.2/layers/layers.0/cross_resolution_weighting/Mul",
            "/model/backbone/stage0/stage0.2/layers/layers.0/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage0/stage0.2/layers/layers.1/cross_resolution_weighting/Mul",
            "/model/backbone/stage0/stage0.2/layers/layers.1/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage0/stage0.2/Add_1",
            "/model/backbone/stage0/stage0.3/layers/layers.0/cross_resolution_weighting/Mul",
            "/model/backbone/stage0/stage0.3/layers/layers.0/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage0/stage0.3/layers/layers.1/cross_resolution_weighting/Mul",
            "/model/backbone/stage0/stage0.3/layers/layers.1/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage0/stage0.3/Add_1",
            "/model/backbone/stage1/stage1.0/layers/layers.0/cross_resolution_weighting/Mul",
            "/model/backbone/stage1/stage1.0/layers/layers.0/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage1/stage1.0/layers/layers.0/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage1/stage1.0/layers/layers.1/cross_resolution_weighting/Mul",
            "/model/backbone/stage1/stage1.0/layers/layers.1/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage1/stage1.0/Add_1",
            "/model/backbone/stage1/stage1.0/layers/layers.1/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage1/stage1.0/Add_2",
            "/model/backbone/stage1/stage1.0/Add_5",
            "/model/backbone/stage1/stage1.1/layers/layers.0/cross_resolution_weighting/Mul",
            "/model/backbone/stage1/stage1.1/layers/layers.0/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage1/stage1.1/layers/layers.0/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage1/stage1.1/layers/layers.1/cross_resolution_weighting/Mul",
            "/model/backbone/stage1/stage1.1/layers/layers.1/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage1/stage1.1/Add_1",
            "/model/backbone/stage1/stage1.1/layers/layers.1/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage1/stage1.1/Add_2",
            "/model/backbone/stage1/stage1.1/Add_5",
            "/model/backbone/stage1/stage1.2/layers/layers.0/cross_resolution_weighting/Mul",
            "/model/backbone/stage1/stage1.2/layers/layers.0/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage1/stage1.2/layers/layers.0/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage1/stage1.2/layers/layers.1/cross_resolution_weighting/Mul",
            "/model/backbone/stage1/stage1.2/layers/layers.1/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage1/stage1.2/Add_1",
            "/model/backbone/stage1/stage1.2/layers/layers.1/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage1/stage1.2/Add_2",
            "/model/backbone/stage1/stage1.2/Add_5",
            "/model/backbone/stage1/stage1.3/layers/layers.0/cross_resolution_weighting/Mul",
            "/model/backbone/stage1/stage1.3/layers/layers.0/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage1/stage1.3/layers/layers.0/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage1/stage1.3/layers/layers.1/cross_resolution_weighting/Mul",
            "/model/backbone/stage1/stage1.3/layers/layers.1/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage1/stage1.3/Add_1",
            "/model/backbone/stage1/stage1.3/layers/layers.1/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage1/stage1.3/Add_2",
            "/model/backbone/stage1/stage1.3/Add_5",
            "/model/aggregator/Add",
            "/model/aggregator/Add_1",
        ]

        return {
            "ignored_scope": {
                "names": ignored_scope_names,
            },
            "preset": "mixed",
        }


class LiteHRNet18(BaseSegmNNModel):
    """LiteHRNet18 Model."""

    @property
    def ignore_scope(self) -> dict[str, str | dict[str, list[str]]]:
        """The ignored scope of the LiteHRNet18 model."""
        ignored_scope_names = [
            "/model/backbone/stage0/stage0.0/layers/layers.0/cross_resolution_weighting/Mul",
            "/model/backbone/stage0/stage0.0/layers/layers.0/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage0/stage0.0/layers/layers.1/cross_resolution_weighting/Mul",
            "/model/backbone/stage0/stage0.0/layers/layers.1/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage0/stage0.0/Add_1",
            "/model/backbone/stage0/stage0.1/layers/layers.0/cross_resolution_weighting/Mul",
            "/model/backbone/stage0/stage0.1/layers/layers.0/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage0/stage0.1/layers/layers.1/cross_resolution_weighting/Mul",
            "/model/backbone/stage0/stage0.1/layers/layers.1/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage0/stage0.1/Add_1",
            "/model/backbone/stage1/stage1.0/layers/layers.0/cross_resolution_weighting/Mul",
            "/model/backbone/stage1/stage1.0/layers/layers.0/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage1/stage1.0/layers/layers.0/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage1/stage1.0/layers/layers.1/cross_resolution_weighting/Mul",
            "/model/backbone/stage1/stage1.0/layers/layers.1/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage1/stage1.0/Add_1",
            "/model/backbone/stage1/stage1.0/layers/layers.1/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage1/stage1.0/Add_2",
            "/model/backbone/stage1/stage1.0/Add_5",
            "/model/backbone/stage1/stage1.1/layers/layers.0/cross_resolution_weighting/Mul",
            "/model/backbone/stage1/stage1.1/layers/layers.0/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage1/stage1.1/layers/layers.0/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage1/stage1.1/layers/layers.1/cross_resolution_weighting/Mul",
            "/model/backbone/stage1/stage1.1/layers/layers.1/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage1/stage1.1/Add_1",
            "/model/backbone/stage1/stage1.1/layers/layers.1/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage1/stage1.1/Add_2",
            "/model/backbone/stage1/stage1.1/Add_5",
            "/model/backbone/stage1/stage1.2/layers/layers.0/cross_resolution_weighting/Mul",
            "/model/backbone/stage1/stage1.2/layers/layers.0/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage1/stage1.2/layers/layers.0/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage1/stage1.2/layers/layers.1/cross_resolution_weighting/Mul",
            "/model/backbone/stage1/stage1.2/layers/layers.1/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage1/stage1.2/Add_1",
            "/model/backbone/stage1/stage1.2/layers/layers.1/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage1/stage1.2/Add_2",
            "/model/backbone/stage1/stage1.2/Add_5",
            "/model/backbone/stage1/stage1.3/layers/layers.0/cross_resolution_weighting/Mul",
            "/model/backbone/stage1/stage1.3/layers/layers.0/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage1/stage1.3/layers/layers.0/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage1/stage1.3/layers/layers.1/cross_resolution_weighting/Mul",
            "/model/backbone/stage1/stage1.3/layers/layers.1/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage1/stage1.3/Add_1",
            "/model/backbone/stage1/stage1.3/layers/layers.1/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage1/stage1.3/Add_2",
            "/model/backbone/stage1/stage1.3/Add_5",
            "/model/backbone/stage2/stage2.0/layers/layers.0/cross_resolution_weighting/Mul",
            "/model/backbone/stage2/stage2.0/layers/layers.0/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage2/stage2.0/layers/layers.0/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage2/stage2.0/layers/layers.0/cross_resolution_weighting/Mul_3",
            "/model/backbone/stage2/stage2.0/layers/layers.1/cross_resolution_weighting/Mul",
            "/model/backbone/stage2/stage2.0/layers/layers.1/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage2/stage2.0/Add_1",
            "/model/backbone/stage2/stage2.0/layers/layers.1/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage2/stage2.0/Add_2",
            "/model/backbone/stage2/stage2.0/layers/layers.1/cross_resolution_weighting/Mul_3",
            "/model/backbone/stage2/stage2.0/Add_3",
            "/model/backbone/stage2/stage2.0/Add_6",
            "/model/backbone/stage2/stage2.0/Add_7",
            "/model/backbone/stage2/stage2.0/Add_11",
            "/model/backbone/stage2/stage2.1/layers/layers.0/cross_resolution_weighting/Mul",
            "/model/backbone/stage2/stage2.1/layers/layers.0/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage2/stage2.1/layers/layers.0/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage2/stage2.1/layers/layers.0/cross_resolution_weighting/Mul_3",
            "/model/backbone/stage2/stage2.1/layers/layers.1/cross_resolution_weighting/Mul",
            "/model/backbone/stage2/stage2.1/layers/layers.1/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage2/stage2.1/Add_1",
            "/model/backbone/stage2/stage2.1/layers/layers.1/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage2/stage2.1/Add_2",
            "/model/backbone/stage2/stage2.1/layers/layers.1/cross_resolution_weighting/Mul_3",
            "/model/backbone/stage2/stage2.1/Add_3",
            "/model/backbone/stage2/stage2.1/Add_6",
            "/model/backbone/stage2/stage2.1/Add_7",
            "/model/backbone/stage2/stage2.1/Add_11",
            "/model/aggregator/Add",
            "/model/aggregator/Add_1",
            "/model/aggregator/Add_2",
            "/model/backbone/stage2/stage2.1/Add",
        ]

        return {
            "ignored_scope": {
                "patterns": ["/model/backbone/*"],
                "names": ignored_scope_names,
            },
            "preset": "mixed",
        }


class LiteHRNetX(BaseSegmNNModel):
    """LiteHRNetX Model."""

    @property
    def ignore_scope(self) -> dict[str, str | dict[str, list[str]]]:
        """The ignored scope of the LiteHRNetX model."""
        ignored_scope_names = [
            "/model/backbone/stage0/stage0.0/layers/layers.0/cross_resolution_weighting/Mul",
            "/model/backbone/stage0/stage0.0/layers/layers.0/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage0/stage0.0/layers/layers.1/cross_resolution_weighting/Mul",
            "/model/backbone/stage0/stage0.0/layers/layers.1/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage0/stage0.0/Add_1",
            "/model/backbone/stage0/stage0.1/layers/layers.0/cross_resolution_weighting/Mul",
            "/model/backbone/stage0/stage0.1/layers/layers.0/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage0/stage0.1/layers/layers.1/cross_resolution_weighting/Mul",
            "/model/backbone/stage0/stage0.1/layers/layers.1/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage0/stage0.1/Add_1",
            "/model/backbone/stage1/stage1.0/layers/layers.0/cross_resolution_weighting/Mul",
            "/model/backbone/stage1/stage1.0/layers/layers.0/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage1/stage1.0/layers/layers.0/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage1/stage1.0/layers/layers.1/cross_resolution_weighting/Mul",
            "/model/backbone/stage1/stage1.0/layers/layers.1/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage1/stage1.0/Add_1",
            "/model/backbone/stage1/stage1.0/layers/layers.1/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage1/stage1.0/Add_2",
            "/model/backbone/stage1/stage1.0/Add_5",
            "/model/backbone/stage1/stage1.1/layers/layers.0/cross_resolution_weighting/Mul",
            "/model/backbone/stage1/stage1.1/layers/layers.0/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage1/stage1.1/layers/layers.0/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage1/stage1.1/layers/layers.1/cross_resolution_weighting/Mul",
            "/model/backbone/stage1/stage1.1/layers/layers.1/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage1/stage1.1/Add_1",
            "/model/backbone/stage1/stage1.1/layers/layers.1/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage1/stage1.1/Add_2",
            "/model/backbone/stage1/stage1.1/Add_5",
            "/model/backbone/stage1/stage1.2/layers/layers.0/cross_resolution_weighting/Mul",
            "/model/backbone/stage1/stage1.2/layers/layers.0/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage1/stage1.2/layers/layers.0/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage1/stage1.2/layers/layers.1/cross_resolution_weighting/Mul",
            "/model/backbone/stage1/stage1.2/layers/layers.1/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage1/stage1.2/Add_1",
            "/model/backbone/stage1/stage1.2/layers/layers.1/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage1/stage1.2/Add_2",
            "/model/backbone/stage1/stage1.2/Add_5",
            "/model/backbone/stage1/stage1.3/layers/layers.0/cross_resolution_weighting/Mul",
            "/model/backbone/stage1/stage1.3/layers/layers.0/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage1/stage1.3/layers/layers.0/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage1/stage1.3/layers/layers.1/cross_resolution_weighting/Mul",
            "/model/backbone/stage1/stage1.3/layers/layers.1/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage1/stage1.3/Add_1",
            "/model/backbone/stage1/stage1.3/layers/layers.1/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage1/stage1.3/Add_2",
            "/model/backbone/stage1/stage1.3/Add_5",
            "/model/backbone/stage2/stage2.0/layers/layers.0/cross_resolution_weighting/Mul",
            "/model/backbone/stage2/stage2.0/layers/layers.0/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage2/stage2.0/layers/layers.0/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage2/stage2.0/layers/layers.0/cross_resolution_weighting/Mul_3",
            "/model/backbone/stage2/stage2.0/layers/layers.1/cross_resolution_weighting/Mul",
            "/model/backbone/stage2/stage2.0/layers/layers.1/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage2/stage2.0/Add_1",
            "/model/backbone/stage2/stage2.0/layers/layers.1/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage2/stage2.0/Add_2",
            "/model/backbone/stage2/stage2.0/layers/layers.1/cross_resolution_weighting/Mul_3",
            "/model/backbone/stage2/stage2.0/Add_3",
            "/model/backbone/stage2/stage2.0/Add_6",
            "/model/backbone/stage2/stage2.0/Add_7",
            "/model/backbone/stage2/stage2.0/Add_11",
            "/model/backbone/stage2/stage2.1/layers/layers.0/cross_resolution_weighting/Mul",
            "/model/backbone/stage2/stage2.1/layers/layers.0/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage2/stage2.1/layers/layers.0/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage2/stage2.1/layers/layers.0/cross_resolution_weighting/Mul_3",
            "/model/backbone/stage2/stage2.1/layers/layers.1/cross_resolution_weighting/Mul",
            "/model/backbone/stage2/stage2.1/layers/layers.1/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage2/stage2.1/Add_1",
            "/model/backbone/stage2/stage2.1/layers/layers.1/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage2/stage2.1/Add_2",
            "/model/backbone/stage2/stage2.1/layers/layers.1/cross_resolution_weighting/Mul_3",
            "/model/backbone/stage2/stage2.1/Add_3",
            "/model/backbone/stage2/stage2.1/Add_6",
            "/model/backbone/stage2/stage2.1/Add_7",
            "/model/backbone/stage2/stage2.1/Add_11",
            "/model/backbone/stage2/stage2.2/layers/layers.0/cross_resolution_weighting/Mul",
            "/model/backbone/stage2/stage2.2/layers/layers.0/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage2/stage2.2/layers/layers.0/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage2/stage2.2/layers/layers.0/cross_resolution_weighting/Mul_3",
            "/model/backbone/stage2/stage2.2/layers/layers.1/cross_resolution_weighting/Mul",
            "/model/backbone/stage2/stage2.2/layers/layers.1/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage2/stage2.2/Add_1",
            "/model/backbone/stage2/stage2.2/layers/layers.1/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage2/stage2.2/Add_2",
            "/model/backbone/stage2/stage2.2/layers/layers.1/cross_resolution_weighting/Mul_3",
            "/model/backbone/stage2/stage2.2/Add_3",
            "/model/backbone/stage2/stage2.2/Add_6",
            "/model/backbone/stage2/stage2.2/Add_7",
            "/model/backbone/stage2/stage2.2/Add_11",
            "/model/backbone/stage2/stage2.3/layers/layers.0/cross_resolution_weighting/Mul",
            "/model/backbone/stage2/stage2.3/layers/layers.0/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage2/stage2.3/layers/layers.0/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage2/stage2.3/layers/layers.0/cross_resolution_weighting/Mul_3",
            "/model/backbone/stage2/stage2.3/layers/layers.1/cross_resolution_weighting/Mul",
            "/model/backbone/stage2/stage2.3/layers/layers.1/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage2/stage2.3/Add_1",
            "/model/backbone/stage2/stage2.3/layers/layers.1/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage2/stage2.3/Add_2",
            "/model/backbone/stage2/stage2.3/layers/layers.1/cross_resolution_weighting/Mul_3",
            "/model/backbone/stage2/stage2.3/Add_3",
            "/model/backbone/stage2/stage2.3/Add_6",
            "/model/backbone/stage2/stage2.3/Add_7",
            "/model/backbone/stage2/stage2.3/Add_11",
            "/model/backbone/stage3/stage3.0/layers/layers.0/cross_resolution_weighting/Mul",
            "/model/backbone/stage3/stage3.0/layers/layers.0/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage3/stage3.0/layers/layers.0/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage3/stage3.0/layers/layers.0/cross_resolution_weighting/Mul_3",
            "/model/backbone/stage3/stage3.0/layers/layers.0/cross_resolution_weighting/Mul_4",
            "/model/backbone/stage3/stage3.0/layers/layers.1/cross_resolution_weighting/Mul",
            "/model/backbone/stage3/stage3.0/layers/layers.1/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage3/stage3.0/Add_1",
            "/model/backbone/stage3/stage3.0/layers/layers.1/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage3/stage3.0/Add_2",
            "/model/backbone/stage3/stage3.0/layers/layers.1/cross_resolution_weighting/Mul_3",
            "/model/backbone/stage3/stage3.0/Add_3",
            "/model/backbone/stage3/stage3.0/layers/layers.1/cross_resolution_weighting/Mul_4",
            "/model/backbone/stage3/stage3.0/Add_4",
            "/model/backbone/stage3/stage3.0/Add_7",
            "/model/backbone/stage3/stage3.0/Add_8",
            "/model/backbone/stage3/stage3.0/Add_9",
            "/model/backbone/stage3/stage3.0/Add_13",
            "/model/backbone/stage3/stage3.0/Add_14",
            "/model/backbone/stage3/stage3.0/Add_19",
            "/model/backbone/stage3/stage3.1/layers/layers.0/cross_resolution_weighting/Mul",
            "/model/backbone/stage3/stage3.1/layers/layers.0/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage3/stage3.1/layers/layers.0/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage3/stage3.1/layers/layers.0/cross_resolution_weighting/Mul_3",
            "/model/backbone/stage3/stage3.1/layers/layers.0/cross_resolution_weighting/Mul_4",
            "/model/backbone/stage3/stage3.1/layers/layers.1/cross_resolution_weighting/Mul",
            "/model/backbone/stage3/stage3.1/layers/layers.1/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage3/stage3.1/Add_1",
            "/model/backbone/stage3/stage3.1/layers/layers.1/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage3/stage3.1/Add_2",
            "/model/backbone/stage3/stage3.1/layers/layers.1/cross_resolution_weighting/Mul_3",
            "/model/backbone/stage3/stage3.1/Add_3",
            "/model/backbone/stage3/stage3.1/layers/layers.1/cross_resolution_weighting/Mul_4",
            "/model/backbone/stage3/stage3.1/Add_4",
            "/model/backbone/stage3/stage3.1/Add_7",
            "/model/backbone/stage3/stage3.1/Add_8",
            "/model/backbone/stage3/stage3.1/Add_9",
            "/model/backbone/stage3/stage3.1/Add_13",
            "/model/backbone/stage3/stage3.1/Add_14",
            "/model/backbone/stage3/stage3.1/Add_19",
            "/model/backbone/stage0/stage0.0/Add",
            "/model/backbone/stage0/stage0.1/Add",
            "/model/backbone/stage1/stage1.0/Add",
            "/model/backbone/stage1/stage1.1/Add",
            "/model/backbone/stage1/stage1.2/Add",
            "/model/backbone/stage1/stage1.3/Add",
            "/model/backbone/stage2/stage2.0/Add",
            "/model/backbone/stage2/stage2.1/Add",
            "/model/backbone/stage2/stage2.2/Add",
            "/model/backbone/stage2/stage2.3/Add",
            "/model/backbone/stage3/stage3.0/Add",
            "/model/backbone/stage3/stage3.1/Add",
        ]

        return {
            "ignored_scope": {
                "patterns": ["/model/aggregator/*"],
                "names": ignored_scope_names,
            },
            "preset": "performance",
        }


LITEHRNET_VARIANTS = {
    "LiteHRNet18": LiteHRNet18,
    "LiteHRNetS": LiteHRNetS,
    "LiteHRNetX": LiteHRNetX,
}


class OTXLiteHRNet(TorchVisionCompatibleModel):
    """LiteHRNet Model."""

    def _create_model(self) -> nn.Module:
        backbone = LiteHRNet(**self.backbone_configuration)
        decode_head = CustomFCNHead(num_classes=self.num_classes, **self.decode_head_configuration)

        return LITEHRNET_VARIANTS[self.name_base_model](
            backbone=backbone,
            decode_head=decode_head,
            criterion_configuration=self.criterion_configuration,
        )

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_seg_lite_hrnet_ckpt(state_dict, add_prefix)

    @property
    def _optimization_config(self) -> dict[str, Any]:
        """PTQ config for LiteHRNet."""
        # TODO(Kirill): check PTQ without adding the whole backbone to ignored_scope
        ignored_scope = self.model.ignore_scope
        optim_config = {
            "advanced_parameters": {
                "activations_range_estimator_params": {
                    "min": {"statistics_type": "QUANTILE", "aggregator_type": "MIN", "quantile_outlier_prob": 1e-4},
                    "max": {"statistics_type": "QUANTILE", "aggregator_type": "MAX", "quantile_outlier_prob": 1e-4},
                },
            },
        }
        optim_config.update(ignored_scope)
        return optim_config
