# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""ATSS model implementations."""

from typing import Literal

from otx.algo.utils.mmconfig import read_mmconfig
from otx.core.model.entity.detection import MMDetCompatibleModel


class ATSS(MMDetCompatibleModel):
    """ATSS Model."""

    def __init__(self, num_classes: int, variant: Literal["mobilenetv2", "r50_fpn", "resnext101"]) -> None:
        model_name = f"atss_{variant}"
        config = read_mmconfig(model_name=model_name)
        super().__init__(num_classes=num_classes, config=config)
