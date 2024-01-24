# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""MaskRCNN model implementations."""

from __future__ import annotations

from copy import copy
from typing import Any, Literal

from otx.algo.utils.mmconfig import read_mmconfig
from otx.core.model.entity.instance_segmentation import MMDetInstanceSegCompatibleModel
from otx.core.utils.utils import get_mean_std_from_data_processing


class MaskRCNN(MMDetInstanceSegCompatibleModel):
    """MaskRCNN Model."""

    def __init__(self, num_classes: int, variant: Literal["efficientnetb2b", "r50"]) -> None:
        model_name = f"maskrcnn_{variant}"
        config = read_mmconfig(model_name=model_name)
        super().__init__(num_classes=num_classes, config=config)

    @property
    def export_params(self) -> dict[str, Any]:
        """Parameters for an exporter."""
        export_params = get_mean_std_from_data_processing(self.config)
        export_params["model_builder"] = self._create_model
        export_params["model_cfg"] = copy(self.config)
        export_params["deploy_cfg"] = "otx.config.mmdeploy.instance_segmentation.maskrcnn"
        export_params["input_size"] = (1, 3, 1024, 1024)
        export_params["resize_mode"] = "fit_to_window"
        export_params["pad_value"] = 0
        export_params["swap_rgb"] = False

        return export_params


class MaskRCNNSwinT(MMDetInstanceSegCompatibleModel):
    """MaskRCNNSwinT Model."""

    def __init__(self, num_classes: int) -> None:
        config = read_mmconfig(model_name="maskrcnn_swint")
        super().__init__(num_classes=num_classes, config=config)

    @property
    def export_params(self) -> dict[str, Any]:
        """Parameters for an exporter."""
        export_params = get_mean_std_from_data_processing(self.config)
        export_params["model_builder"] = self._create_model
        export_params["model_cfg"] = copy(self.config)
        export_params["deploy_cfg"] = "otx.config.mmdeploy.instance_segmentation.maskrcnn_swint"
        export_params["input_size"] = (1, 3, 1344, 1344)
        export_params["resize_mode"] = "fit_to_window"
        export_params["pad_value"] = 0
        export_params["swap_rgb"] = False

        return export_params
