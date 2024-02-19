# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for visual prompting model entity used in OTX."""

from __future__ import annotations

from typing import Any

from otx.core.data.entity.base import T_OTXBatchPredEntityWithXAI
from otx.core.data.entity.tile import T_OTXTileBatchDataEntity
from otx.core.data.entity.visual_prompting import (
    VisualPromptingBatchDataEntity,
    VisualPromptingBatchPredEntity,
    ZeroShotVisualPromptingBatchDataEntity,
    ZeroShotVisualPromptingBatchPredEntity,
)
from otx.core.model.entity.base import OTXModel


class OTXVisualPromptingModel(
    OTXModel[
        VisualPromptingBatchDataEntity,
        VisualPromptingBatchPredEntity,
        T_OTXBatchPredEntityWithXAI,
        T_OTXTileBatchDataEntity,
    ],
):
    """Base class for the visual prompting models used in OTX."""

    def __init__(self, num_classes: int = 0) -> None:
        super().__init__(num_classes=num_classes)


class OTXZeroShotVisualPromptingModel(
    OTXModel[
        ZeroShotVisualPromptingBatchDataEntity,
        ZeroShotVisualPromptingBatchPredEntity,
        T_OTXBatchPredEntityWithXAI,
        T_OTXTileBatchDataEntity,
    ],
):
    """Base class for the zero-shot visual prompting models used in OTX."""

    def __init__(self, num_classes: int = 0) -> None:
        super().__init__(num_classes=num_classes)

        self._register_load_state_dict_pre_hook(self.load_state_dict_pre_hook)

    def state_dict(
        self,
        *args,
        destination: dict[str, Any] | None = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> dict[str, Any] | None:
        """Return state dictionary of model entity with reference features, masks, and used indices."""
        super().state_dict(*args, destination=destination, prefix=prefix, keep_vars=keep_vars)

        if isinstance(destination, dict):
            # to save reference_info instead of reference_feats only
            destination.pop(prefix + "model.reference_info.reference_feats")
            destination.update({prefix + "model.reference_info": self.model.reference_info})
        return destination

    def load_state_dict_pre_hook(self, state_dict: dict[str, Any], prefix: str = "", *args, **kwargs) -> None:
        """Load reference info manually."""
        self.model.reference_info = state_dict.get(prefix + "model.reference_info", self.model.reference_info)
        state_dict[prefix + "model.reference_info.reference_feats"] = self.model.reference_info["reference_feats"]
