# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for visual prompting model entity used in OTX."""

from __future__ import annotations

from typing import TYPE_CHECKING

from otx.core.data.entity.visual_prompting import VisualPromptingBatchDataEntity, VisualPromptingBatchPredEntity
from otx.core.model.entity.base import OTXModel
from otx.core.utils.config import inplace_num_classes

if TYPE_CHECKING:
    from omegaconf import DictConfig


class OTXVisualPromptingModel(
    OTXModel[VisualPromptingBatchDataEntity, VisualPromptingBatchPredEntity],
):
    """Base class for the visual prompting models used in OTX."""

    def __init__(self, num_classes: int = 0) -> None:
        super().__init__(num_classes=num_classes)
