# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for visual prompting model entity used in OTX."""

from __future__ import annotations

from otx.core.data.entity.tile import T_OTXTileBatchDataEntity
from otx.core.data.entity.visual_prompting import VisualPromptingBatchDataEntity, VisualPromptingBatchPredEntity, ZeroShotVisualPromptingBatchDataEntity, ZeroShotVisualPromptingBatchPredEntity
from otx.core.model.entity.base import OTXModel


class OTXVisualPromptingModel(
    OTXModel[VisualPromptingBatchDataEntity, VisualPromptingBatchPredEntity, T_OTXTileBatchDataEntity],
):
    """Base class for the visual prompting models used in OTX."""

    def __init__(self, num_classes: int = 0) -> None:
        super().__init__(num_classes=num_classes)
        
        
class OTXZeroShotVisualPromptingModel(
    OTXModel[ZeroShotVisualPromptingBatchDataEntity, ZeroShotVisualPromptingBatchPredEntity, T_OTXTileBatchDataEntity],
):
    """Base class for the zero-shot visual prompting models used in OTX."""

    def __init__(self, num_classes: int = 0) -> None:
        super().__init__(num_classes=num_classes)
