# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""RTMDetInst model implementations."""

from __future__ import annotations

from typing import Any, Literal

from otx.algo.utils.mmconfig import read_mmconfig
from otx.core.model.entity.instance_segmentation import MMDetInstanceSegCompatibleModel


class RTMDetInst(MMDetInstanceSegCompatibleModel):
    """RTMDetInst Model."""

    def __init__(self, num_classes: int, variant: Literal["tiny"]) -> None:
        model_name = f"rtmdet_inst_{variant}"
        config = read_mmconfig(model_name=model_name)
        super().__init__(num_classes=num_classes, config=config)
        self.image_size = (1, 3, 640, 640)
        self.tile_image_size = self.image_size

    @property
    def _export_parameters(self) -> dict[str, Any]:
        """Parameters for an exporter."""
        export_params = super()._export_parameters
        export_params["deploy_cfg"] = "otx.algo.instance_segmentation.rtmdet_inst.RTMDetInst"
        export_params["input_size"] = self.image_size
        export_params["resize_mode"] = "fit_to_window_letterbox"
        export_params["pad_value"] = 114
        export_params["swap_rgb"] = False

        # TODO: Add more export parameters if necessary
        # codebase_config = dict(
        #     post_processing=dict(
        #         max_output_boxes_per_class=100,
        #         pre_top_k=300,
        #     )
        # )

        return export_params
