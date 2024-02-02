# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""YOLOX model implementations."""

from __future__ import annotations

from typing import Any, Literal

from otx.algo.utils.mmconfig import read_mmconfig
from otx.algo.utils.support_otx_v1 import OTXv1Helper
from otx.core.model.entity.detection import MMDetCompatibleModel


class YoloX(MMDetCompatibleModel):
    """YoloX Model."""

    def __init__(self, num_classes: int, variant: Literal["l", "s", "tiny", "x"]) -> None:
        model_name = f"yolox_{variant}"
        config = read_mmconfig(model_name=model_name)
        super().__init__(num_classes=num_classes, config=config)
        self.image_size = (1, 3, 640, 640)

    @property
    def _export_parameters(self) -> dict[str, Any]:
        """Parameters for an exporter."""
        export_params = super()._export_parameters
        export_params["resize_mode"] = "fit_to_window_letterbox"
        export_params["pad_value"] = 114
        export_params["swap_rgb"] = True
        export_params["input_size"] = self.image_size
        export_params["deploy_cfg"] = "otx.algo.detection.mmdeploy.yolox"

        return export_params

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_det_ckpt(state_dict, add_prefix)


class YoloXTiny(YoloX):
    """YoloX tiny Model."""

    def __init__(self, num_classes: int) -> None:
        super().__init__(num_classes=num_classes, variant="tiny")
        self.image_size = (1, 3, 416, 416)

    @property
    def _export_parameters(self) -> dict[str, Any]:
        """Parameters for an exporter."""
        export_params = super()._export_parameters
        export_params["resize_mode"] = "fit_to_window_letterbox"
        export_params["pad_value"] = 114
        export_params["swap_rgb"] = False
        export_params["input_size"] = self.image_size
        export_params["deploy_cfg"] = "otx.algo.detection.mmdeploy.yolox_tiny"

        return export_params
