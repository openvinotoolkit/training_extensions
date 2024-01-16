# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""X3D model implementation."""

from otx.algo.utils.mmconfig import read_mmconfig
from otx.core.model.entity.action_classification import MMActionCompatibleModel


class X3D(MMActionCompatibleModel):
    """X3D Model."""

    def __init__(self, num_classes: int) -> None:
        config = read_mmconfig("x3d")
        super().__init__(num_classes=num_classes, config=config)
