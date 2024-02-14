# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for rotated detection lightning module used in OTX."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from otx.algo.instance_segmentation.otx_instseg_evaluation import (
    OTXMaskRLEMeanAveragePrecision,
)
from otx.core.model.entity.rotated_detection import OTXRotatedDetModel
from otx.core.model.module.instance_segmentation import OTXInstanceSegLitModule

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable


class OTXRotatedDetLitModule(OTXInstanceSegLitModule):
    """Base class for the lightning module used in OTX rotated detection task."""

    def __init__(
        self,
        otx_model: OTXRotatedDetModel,
        torch_compile: bool,
        optimizer: list[OptimizerCallable] | OptimizerCallable = lambda p: torch.optim.SGD(p, lr=0.01),
        scheduler: list[LRSchedulerCallable] | LRSchedulerCallable = torch.optim.lr_scheduler.ConstantLR,
    ):
        super().__init__(
            otx_model=otx_model,
            torch_compile=torch_compile,
            optimizer=optimizer,
            scheduler=scheduler,
        )

        self.val_metric = OTXMaskRLEMeanAveragePrecision(iou_type="segm")
        self.test_metric = OTXMaskRLEMeanAveragePrecision(iou_type="segm")
