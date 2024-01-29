# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for visual prompting model entity used in OTX."""

from __future__ import annotations

from otx.core.data.entity.tile import T_OTXTileBatchDataEntity
from otx.core.data.entity.visual_prompting import (
    VisualPromptingBatchDataEntity,
    VisualPromptingBatchPredEntity,
    ZeroShotVisualPromptingBatchDataEntity,
    ZeroShotVisualPromptingBatchPredEntity,
)
from otx.core.model.entity.base import OTXModel
from typing import Any


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
        
    def state_dict(self) -> dict[str, Any]:
        """Return state dictionary of model entity with reference features, masks, and used indices."""
        state_dict = super().state_dict()
        state_dict.update(
            {
                "model.model.reference_feats": self.model.model.reference_feats,
                "model.model.reference_masks": self.model.model.reference_masks,
                "model.model.used_indices": self.model.model.used_indices,
            },
        )
        return state_dict
    
    def load_state_dict(self, ckpt: dict[str, Any], *args, **kwargs) -> None:
        """Load state dictionary from checkpoint state dictionary."""
        ckpt_meta_info = ckpt.pop("meta_info", None)  # noqa: F841

        self.model.model.reference_feats = ckpt.pop("model.model.reference_feats").to(self.device)
        self.model.model.reference_masks = [m.to(self.device) for m in ckpt.pop("model.model.reference_masks")]
        self.model.model.used_indices = ckpt.pop("model.model.used_indices")
        return super().load_state_dict(ckpt, *args, **kwargs)
