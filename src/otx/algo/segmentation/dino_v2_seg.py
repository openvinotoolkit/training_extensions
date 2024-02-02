# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""DinoV2Seg model implementations."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from otx.algo.utils.mmconfig import read_mmconfig
from otx.core.model.entity.segmentation import MMSegCompatibleModel


class DinoV2Seg(MMSegCompatibleModel):
    """DinoV2Seg Model."""

    def __init__(self, num_classes: int) -> None:
        model_name = "dino_v2_seg"
        config = read_mmconfig(model_name=model_name)
        super().__init__(num_classes=num_classes, config=config)

    @property
    def _export_parameters(self) -> dict[str, Any]:
        """Defines parameters required to export a particular model implementation."""
        parent_parameters = super()._export_parameters
        parent_parameters["input_size"] = (1, 3, 560, 560)
        return parent_parameters

    @property
    def _optimization_config(self) -> dict[str, Any]:
        """PTQ config for DinoV2Seg."""
        return {"model_type": "transformer"}
