# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""LiteHRNet model implementations."""

from typing import Literal

from otx.algo.utils.mmconfig import read_mmconfig
from otx.algo.utils.support_otx_v1 import OTXv1Helper
from otx.core.model.entity.segmentation import MMSegCompatibleModel


class LiteHRNet(MMSegCompatibleModel):
    """LiteHRNet Model."""

    def __init__(self, num_classes: int, variant: Literal["18", 18, "s", "x"]) -> None:
        model_name = f"litehrnet_{variant}"
        config = read_mmconfig(model_name=model_name)
        super().__init__(num_classes=num_classes, config=config)

    def load_from_otx_v1_ckpt(self, state_dict, add_prefix: str="model.model."):
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_seg_lite_hrnet_ckpt(state_dict, add_prefix) 