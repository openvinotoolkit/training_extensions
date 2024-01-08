# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for visual prompting model entity used in OTX."""

from __future__ import annotations

from otx.core.data.entity.visual_prompting import (
    VisualPromptingBatchDataEntity, VisualPromptingBatchPredEntity)
from otx.core.model.entity.base import OTXModel


class OTXVisualPromptingModel(
    OTXModel[VisualPromptingBatchDataEntity, VisualPromptingBatchPredEntity],
):
    """Base class for the visual prompting models used in OTX."""
