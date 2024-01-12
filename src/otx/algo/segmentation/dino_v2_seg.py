# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""DinoV2Seg model implementations."""

from otx.algo.utils.mmconfig import read_mmconfig
from otx.core.model.entity.segmentation import MMSegCompatibleModel


class DinoV2Seg(MMSegCompatibleModel):
    """DinoV2Seg Model."""

    def __init__(self, num_classes: int) -> None:
        model_name = "dino_v2_seg"
        config = read_mmconfig(model_name=model_name)
        super().__init__(num_classes=num_classes, config=config)
