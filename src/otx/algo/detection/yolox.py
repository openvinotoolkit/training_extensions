# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""YOLOX model implementations."""

from typing import Literal

from otx.algo.utils.mmconfig import read_mmconfig
from otx.algo.utils.support_otx_v1 import OTXv1Helper
from otx.core.model.entity.detection import MMDetCompatibleModel


class YoloX(MMDetCompatibleModel):
    """YoloX Model."""

    def __init__(self, num_classes: int, variant: Literal["l", "s", "tiny", "x"]) -> None:
        model_name = f"yolox_{variant}"
        config = read_mmconfig(model_name=model_name)
        super().__init__(num_classes=num_classes, config=config)

    def load_from_otx_v1_ckpt(self, state_dict, add_prefix: str="model.model."):
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_det_ckpt(state_dict, add_prefix) 