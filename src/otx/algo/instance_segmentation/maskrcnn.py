# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""MaskRCNN model implementations."""

from __future__ import annotations

from typing import Any, Literal

from otx.algo.utils.mmconfig import read_mmconfig
from otx.algo.utils.support_otx_v1 import OTXv1Helper
from otx.core.model.entity.instance_segmentation import MMDetInstanceSegCompatibleModel


class MaskRCNN(MMDetInstanceSegCompatibleModel):
    """MaskRCNN Model."""

    def __init__(self, num_classes: int, variant: Literal["efficientnetb2b", "r50", "swint"]) -> None:
        model_name = f"maskrcnn_{variant}"
        config = read_mmconfig(model_name=model_name)
        super().__init__(num_classes=num_classes, config=config)
        self.image_size = (1, 3, 1024, 1024)
        self.tile_image_size = (1, 3, 512, 512)

    @property
    def _export_parameters(self) -> dict[str, Any]:
        """Parameters for an exporter."""
        export_params = super()._export_parameters
        export_params["deploy_cfg"] = "otx.algo.instance_segmentation.mmdeploy.maskrcnn"
        export_params["input_size"] = self.image_size
        export_params["resize_mode"] = "standard"  # [TODO](@Eunwoo): need to revert it to fit_to_window after resolving
        export_params["pad_value"] = 0
        export_params["swap_rgb"] = False

        return export_params

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_iseg_ckpt(state_dict, add_prefix)


class MaskRCNNSwinT(MaskRCNN):
    """MaskRCNNSwinT Model."""

    def __init__(self, num_classes: int) -> None:
        super().__init__(num_classes=num_classes, variant="swint")
        self.image_size = (1, 3, 1344, 1344)
        self.tile_image_size = (1, 3, 512, 512)

    @property
    def _export_parameters(self) -> dict[str, Any]:
        """Parameters for an exporter."""
        export_params = super()._export_parameters
        export_params["deploy_cfg"] = "otx.algo.instance_segmentation.mmdeploy.maskrcnn_swint"
        export_params["input_size"] = self.image_size
        export_params["resize_mode"] = "standard"  # [TODO](@Eunwoo): need to revert it to fit_to_window after resolving
        export_params["pad_value"] = 0
        export_params["swap_rgb"] = False

        return export_params
