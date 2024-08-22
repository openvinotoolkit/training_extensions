# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Class definition for diffusion model entity used in OTX."""
from __future__ import annotations

from typing import TYPE_CHECKING

from otx.core.data.entity.diffusion import (
    DiffusionBatchDataEntity,
    DiffusionBatchPredEntity,
)
from otx.core.model.base import OTXModel

if TYPE_CHECKING:
    from torch import nn


class OTXDiffusionModel(OTXModel[DiffusionBatchDataEntity, DiffusionBatchPredEntity]):
    """OTX Diffusion model."""

    def _create_model(self) -> nn.Module:
        raise NotImplementedError
