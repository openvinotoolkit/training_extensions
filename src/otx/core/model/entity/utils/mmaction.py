# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for classification model entity used in OTX."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mmaction.models.data_preprocessors import (
    ActionDataPreprocessor as _ActionDataPreprocessor,
)
from mmaction.registry import MODELS
from mmengine.registry import MODELS as MMENGINE_MODELS

from otx.core.utils.build import build_mm_model, get_classification_layers

if TYPE_CHECKING:
    from torch import device, nn
    from omegaconf import DictConfig


# NOTE: For the history of this monkey patching, please see
# https://github.com/openvinotoolkit/training_extensions/issues/2743
@MMENGINE_MODELS.register_module(force=True)
class ActionDataPreprocessor(_ActionDataPreprocessor):
    @property
    def device(self) -> device:
        try:
            buf = next(self.buffers())
        except StopIteration:
            return super().device
        else:
            return buf.device


def create_model(config: DictConfig, load_from: str) -> tuple[nn.Module, dict[str, dict[str, int]]]:
    classification_layers = get_classification_layers(config, MODELS, "model.")
    return build_mm_model(config, MODELS, load_from), classification_layers
