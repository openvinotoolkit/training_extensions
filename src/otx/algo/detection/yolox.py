# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""YOLOX model implementations."""

from __future__ import annotations

from copy import copy
from typing import Literal, Any

from otx.algo.utils.mmconfig import read_mmconfig
from otx.core.utils.utils import get_mean_std_from_data_processing
from otx.core.model.entity.detection import MMDetCompatibleModel


class YoloX(MMDetCompatibleModel):
    """YoloX Model."""

    def __init__(self, num_classes: int, variant: Literal["l", "s", "x"]) -> None:
        model_name = f"yolox_{variant}"
        config = read_mmconfig(model_name=model_name)
        super().__init__(num_classes=num_classes, config=config)

    @property
    def export_params(self) -> dict[str, Any]:
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
