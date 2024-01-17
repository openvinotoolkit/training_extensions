# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""YOLOX model implementations."""

from typing import Literal

from otx.algo.utils.mmconfig import read_mmconfig
from otx.core.model.entity.detection import MMDetCompatibleModel


class YoloX(MMDetCompatibleModel):
    """YoloX Model."""

    def __init__(self, num_classes: int, variant: Literal["l", "s", "tiny", "x"]) -> None:
        model_name = f"yolox_{variant}"
        config = read_mmconfig(model_name=model_name)
        super().__init__(num_classes=num_classes, config=config)
