# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""YOLOX model implementations."""

from __future__ import annotations

from copy import copy
from typing import Literal, Any

from otx.algo.utils.mmconfig import read_mmconfig
from otx.core.model.utils import get_mean_std_from_data_processing
from otx.core.model.entity.detection import MMDetCompatibleModel


class YoloX(MMDetCompatibleModel):
    """YoloX Model."""

    def __init__(self, num_classes: int, variant: Literal["l", "s", "tiny", "x"]) -> None:
        self.model_name = f"yolox_{variant}"
        config = read_mmconfig(model_name=self.model_name)
        super().__init__(num_classes=num_classes, config=config)

    def _get_export_parameters(self) -> dict[str, Any]:
        export_params = get_mean_std_from_data_processing(self.config)
        export_params["resize_mode"] = "standard"
        export_params["pad_value"] = 0
        export_params["swap_rgb"] = False
        export_params["via_onnx"] = False

        if self.model_name == "tiny":  # yolox_tiny
            export_params["input_size"] = (1, 3, 416, 416)
            export_params["mmdeploy_config"] = "otx.config.mmdeploy.detection.yolox_tiny"
        else:
            export_params["input_size"] = (1, 3, 640, 640)
            export_params["mmdeploy_config"] = "otx.config.mmdeploy.detection.yolox"

        export_params["mm_model_config"] = copy(self.config)

        return export_params
