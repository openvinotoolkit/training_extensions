# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""RTMDetTiny model implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from otx.algo.utils.mmconfig import read_mmconfig
from otx.algo.utils.support_otx_v1 import OTXv1Helper
from otx.core.metrics.mean_ap import MeanAPCallable
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.core.model.detection import MMDetCompatibleModel

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from otx.core.metrics import MetricCallable


class RTMDet(MMDetCompatibleModel):
    """RTMDet Model."""

    def __init__(
        self,
        num_classes: int,
        variant: Literal["tiny"],
        optimizer: list[OptimizerCallable] | OptimizerCallable = DefaultOptimizerCallable,
        scheduler: list[LRSchedulerCallable] | LRSchedulerCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MeanAPCallable,
        torch_compile: bool = False,
    ) -> None:
        model_name = f"rtmdet_{variant}"
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
        export_params["deploy_cfg"] = "otx.algo.detection.mmdeploy.rtmdet"
        export_params["input_size"] = self.image_size
        export_params["resize_mode"] = "fit_to_window_letterbox"
        export_params["pad_value"] = 114
        export_params["swap_rgb"] = False

        return export_params

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_det_ckpt(state_dict, add_prefix)
