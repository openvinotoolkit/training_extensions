# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX visual prompting data entities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torchvision import tv_tensors

from otx.core.data.entity.base import (OTXBatchDataEntity, OTXBatchPredEntity,
                                       OTXDataEntity, OTXPredEntity)
from otx.core.data.entity.utils import register_pytree_node
from otx.core.types.task import OTXTaskType

if TYPE_CHECKING:
    from torch import LongTensor


@register_pytree_node
@dataclass
class VisualPromptingDataEntity(OTXDataEntity):
    """Data entity for visual prompting task.

    :param labels: labels as integer indices
    """

    @property
    def task(self) -> OTXTaskType:
        """OTX Task type definition."""
        return OTXTaskType.VISUAL_PROMPTING

    labels: LongTensor
    
    
@dataclass
class VisualPromptingPredEntity(VisualPromptingDataEntity, OTXPredEntity):
    """Data entity to represent the visual prompting model output prediction."""
    
    
@dataclass
class VisualPromptingBatchDataEntity(OTXBatchDataEntity[VisualPromptingDataEntity]):
    """Data entity for visual prompting task.

    :param labels: A list of bbox labels as integer indices
    """

    labels: list[LongTensor]

    @property
    def task(self) -> OTXTaskType:
        """OTX Task type definition."""
        return OTXTaskType.VISUAL_PROMPTING

    @classmethod
    def collate_fn(
        cls,
        entities: list[VisualPromptingDataEntity],
    ) -> VisualPromptingBatchDataEntity:
        """Collection function to collect `OTXDataEntity` into `OTXBatchDataEntity` in data loader."""
        batch_data = super().collate_fn(entities)
        return VisualPromptingBatchDataEntity(
            batch_size=batch_data.batch_size,
            images=tv_tensors.Image(data=torch.stack(batch_data.images, dim=0)),
            imgs_info=batch_data.imgs_info,
            labels=[entity.labels for entity in entities],
        )

    def pin_memory(self) -> VisualPromptingBatchDataEntity:
        """Pin memory for member tensor variables."""
        super().pin_memory()
        self.labels = [label.pin_memory() for label in self.labels]
        return self
    
    
@dataclass
class VisualPromptingBatchPredEntity(VisualPromptingBatchDataEntity, OTXBatchPredEntity):
    """Data entity to represent model output predictions for visual prompting task."""