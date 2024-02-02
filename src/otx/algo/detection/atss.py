# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""ATSS model implementations."""

from __future__ import annotations

from typing import Any, Literal

from otx.algo.utils.mmconfig import read_mmconfig
from otx.algo.utils.support_otx_v1 import OTXv1Helper
from otx.core.model.entity.detection import MMDetCompatibleModel


class ATSS(MMDetCompatibleModel):
    """ATSS Model."""

    def __init__(self, num_classes: int, variant: Literal["mobilenetv2", "r50_fpn", "resnext101"]) -> None:
        model_name = f"atss_{variant}"
        config = read_mmconfig(model_name=model_name)
        super().__init__(num_classes=num_classes, config=config)

    @property
    def _export_parameters(self) -> dict[str, Any]:
        """Parameters for an exporter."""
        export_params = super()._export_parameters
        export_params["deploy_cfg"] = "otx.algo.detection.mmdeploy.atss"
        export_params["input_size"] = (1, 3, 736, 992)
        export_params["resize_mode"] = "standard"
        export_params["pad_value"] = 0
        export_params["swap_rgb"] = False

        return export_params

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_det_ckpt(state_dict, add_prefix)


class ATSSR50FPN(ATSS):
    """ATSSR50FPN Model."""

    def __init__(self, num_classes: int) -> None:
        super().__init__(num_classes=num_classes, variant="r50_fpn")

    @property
    def _export_parameters(self) -> dict[str, Any]:
        """Parameters for an exporter."""
        export_params = super()._export_parameters
        export_params["deploy_cfg"] = "otx.algo.detection.mmdeploy.atss_r50_fpn"
        export_params["input_size"] = (1, 3, 800, 1333)
        export_params["resize_mode"] = "fit_to_window"
        export_params["pad_value"] = 0
        export_params["swap_rgb"] = False

        return export_params
