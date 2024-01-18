# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""ATSS model implementations."""

from __future__ import annotations

from copy import copy
from typing import Literal, Any

from otx.algo.utils.mmconfig import read_mmconfig
from otx.core.model.utils import get_mean_std_from_data_processing
from otx.core.model.entity.instance_segmentation import MMDetInstanceSegCompatibleModel


class MaskRCNN(MMDetInstanceSegCompatibleModel):
    """MaskRCNN Model."""

    def __init__(self, num_classes: int, variant: Literal["efficientnetb2b", "r50", "swint"]) -> None:
        self.model_name = f"maskrcnn_{variant}"
        config = read_mmconfig(model_name=self.model_name)
        super().__init__(num_classes=num_classes, config=config)

    def _get_export_parameters(self) -> dict[str, Any]:
        export_params = get_mean_std_from_data_processing(self.config)
        export_params["resize_mode"] = "standard"
        export_params["pad_value"] = 0
        export_params["swap_rgb"] = False
        export_params["via_onnx"] = False

        if self.model_name in ["efficientnetb2b", "r50"]:
            export_params["input_size"] = (1, 3, 1024, 1024)
            export_params["mmdeploy_config"] = "otx.config.mmdeploy.instance_segmentation.maskrcnn"
        elif self.model_name == "swint":
            export_params["input_size"] = (1, 3, 1344, 1344)
            export_params["mmdeploy_config"] = "otx.config.mmdeploy.instance_segmentation.maskrcnn_swint"
        else:
            raise ValueError("Unknown model. Setting mmdeploy is failed.")

        export_params["mm_model_config"] = copy(self.config)
        export_params["mm_model_config"]["load_from"] = self.load_from

        return export_params
