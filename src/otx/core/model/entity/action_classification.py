# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for action_classification model entity used in OTX."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from otx.core.data.entity.action import ActionClsBatchDataEntity, ActionClsBatchPredEntity
from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.model.entity.base import OTXModel
from otx.core.utils.build import build_mm_model

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torch import nn


class OTXActionClsModel(OTXModel[ActionClsBatchDataEntity, ActionClsBatchPredEntity]):
    """Base class for the action classification models used in OTX."""


class MMActionCompitibleModel(OTXActionClsModel):
    """Action classification model compitible for MMAction.

    It can consume MMAction model configuration translated into OTX configuration
    (please see otx.tools.translate_mmrecipe) and create the OTX Action classification model
    compatible for OTX pipelines.
    """

    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.load_from = config.pop("load_from", None)
        super().__init__()

    def _create_model(self) -> nn.Module:
        from mmaction.registry import MODELS

        return build_mm_model(self.config, MODELS, self.load_from)

    def _customize_inputs(self, entity: ActionClsBatchDataEntity) -> dict[str, Any]:
        raise NotImplementedError

    def _customize_outputs(
        self,
        outputs: Any,  # noqa: ANN401
        inputs: ActionClsBatchDataEntity,
    ) -> ActionClsBatchPredEntity | OTXBatchLossEntity:
        raise NotImplementedError
