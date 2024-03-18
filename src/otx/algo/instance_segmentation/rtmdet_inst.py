# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""RTMDetInst model implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from otx.algo.utils.mmconfig import read_mmconfig
from otx.core.metrics.mean_ap import MaskRLEMeanAPCallable
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.core.model.instance_segmentation import MMDetInstanceSegCompatibleModel

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from otx.core.metrics import MetricCallable


class RTMDetInst(MMDetInstanceSegCompatibleModel):
    """RTMDetInst Model."""

    def __init__(
        self,
        num_classes: int,
        variant: Literal["tiny"],
        optimizer: list[OptimizerCallable] | OptimizerCallable = DefaultOptimizerCallable,
        scheduler: list[LRSchedulerCallable] | LRSchedulerCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MaskRLEMeanAPCallable,
        torch_compile: bool = False,
    ) -> None:
        model_name = f"rtmdet_inst_{variant}"
        config = read_mmconfig(model_name=model_name)
        super().__init__(
            num_classes=num_classes,
            config=config,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )
        self.image_size = (1, 3, 640, 640)
        self.tile_image_size = self.image_size

    @property
    def _export_parameters(self) -> dict[str, Any]:
        """Parameters for an exporter."""
        export_params = super()._export_parameters
        export_params["deploy_cfg"] = "otx.algo.instance_segmentation.mmdeploy.rtmdet_inst"
        export_params["input_size"] = self.image_size
        export_params["resize_mode"] = "fit_to_window_letterbox"
        export_params["pad_value"] = 114
        export_params["swap_rgb"] = False

        return export_params
