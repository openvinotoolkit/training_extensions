# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""SegNext model implementations."""

from typing import Literal

from otx.algo.utils.mmconfig import read_mmconfig
from otx.core.model.entity.segmentation import MMSegCompatibleModel


class SegNext(MMSegCompatibleModel):
    """SegNext Model."""

    def __init__(self, num_classes: int, variant: Literal["b", "s", "t"]) -> None:
        model_name = f"segnext_{variant}"
        config = read_mmconfig(model_name=model_name)
        self.image_size = config["data_preprocessor"].get('size', (512,512))
        super().__init__(num_classes=num_classes, config=config)


    def _configure_export_parameters(self) -> None:
        super()._configure_export_parameters()
        self.export_params["via_onnx"] = True
