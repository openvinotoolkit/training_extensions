# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""RTMDetTiny model implementations."""

from copy import copy
from typing import Any, Literal

from otx.algo.utils.mmconfig import read_mmconfig
from otx.core.model.entity.detection import MMDetCompatibleModel
from otx.core.utils.utils import get_mean_std_from_data_processing


class RTMDet(MMDetCompatibleModel):
    """RTMDet Model."""

    def __init__(self, num_classes: int, variant: Literal["tiny"]) -> None:
        model_name = f"rtmdet_{variant}"
        config = read_mmconfig(model_name=model_name)
        super().__init__(num_classes=num_classes, config=config)

    @property
    def export_params(self) -> dict[str, Any]:
        """Parameters for an exporter."""
        export_params = get_mean_std_from_data_processing(self.config)
        export_params["model_builder"] = self._create_model
        export_params["model_cfg"] = copy(self.config)
        export_params["deploy_cfg"] = "otx.config.mmdeploy.detection.rtmdet"
        export_params["input_size"] = (1, 3, 640, 640)
        export_params["resize_mode"] = "fit_to_window_letterbox"
        export_params["pad_value"] = 114
        export_params["swap_rgb"] = False

        return export_params
