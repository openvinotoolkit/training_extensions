# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""YOLOX model implementations."""

from __future__ import annotations

from copy import copy
from typing import Any, Literal

from otx.algo.utils.mmconfig import read_mmconfig
from otx.algo.utils.support_otx_v1 import OTXv1Helper
from otx.core.model.entity.detection import MMDetCompatibleModel
from otx.core.utils.utils import get_mean_std_from_data_processing


class YoloX(MMDetCompatibleModel):
    """YoloX Model."""

    def __init__(self, num_classes: int, variant: Literal["l", "s", "x"]) -> None:
        model_name = f"yolox_{variant}"
        config = read_mmconfig(model_name=model_name)
        super().__init__(num_classes=num_classes, config=config)

    @property
    def _export_parameters(self) -> dict[str, Any]:
        """Parameters for an exporter."""
        export_params = get_mean_std_from_data_processing(self.config)
        export_params["model_builder"] = self._create_model
        export_params["model_cfg"] = copy(self.config)
        export_params["resize_mode"] = "fit_to_window"
        export_params["pad_value"] = 114
        export_params["swap_rgb"] = False
        export_params["input_size"] = (1, 3, 640, 640)
        export_params["deploy_cfg"] = "otx.config.mmdeploy.detection.yolox"

        return export_params

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_det_ckpt(state_dict, add_prefix)


class YoloXTiny(YoloX):
    """YoloX tiny Model."""

    def __init__(self, num_classes: int) -> None:
        super().__init__(num_classes=num_classes, variant="tiny")

    @property
    def _export_parameters(self) -> dict[str, Any]:
        """Parameters for an exporter."""
        export_params = get_mean_std_from_data_processing(self.config)
        export_params["model_builder"] = self._create_model
        export_params["model_cfg"] = copy(self.config)
        export_params["resize_mode"] = "fit_to_window"
        export_params["pad_value"] = 114
        export_params["swap_rgb"] = False
        export_params["input_size"] = (1, 3, 416, 416)
        export_params["deploy_cfg"] = "otx.config.mmdeploy.detection.yolox_tiny"

        return export_params
