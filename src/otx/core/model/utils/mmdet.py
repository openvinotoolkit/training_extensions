# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Utility files for the mmdet package."""

from __future__ import annotations

from typing import TYPE_CHECKING

# TODO(Eugene): Remove this import after the issue is resolved.
from mmdet.models.data_preprocessors import (
    DetDataPreprocessor as _DetDataPreprocessor,
)
from mmengine.registry import MODELS as MMENGINE_MODELS

from otx.core.utils.build import build_mm_model, get_classification_layers

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torch import device, nn


@MMENGINE_MODELS.register_module(force=True)
class DetDataPreprocessor(_DetDataPreprocessor):
    """Class for monkey patching preprocessor.

    NOTE: For the history of this monkey patching, please see
    https://github.com/openvinotoolkit/training_extensions/issues/2743
    """

    @property
    def device(self) -> device:
        """Which device being used."""
        try:
            buf = next(self.buffers())
        except StopIteration:
            return super().device
        else:
            return buf.device


def create_model(config: DictConfig, load_from: str | None) -> tuple[nn.Module, dict[str, dict[str, int]]]:
    """Create a model from mmdet Model registry.

    Args:
        config (DictConfig): Model configuration.
        load_from (str | None): Model weight file path.

    Returns:
        tuple[nn.Module, dict[str, dict[str, int]]]: Model instance and classification layers.
    """
    # classification_layers = get_classification_layers(config, MODELS, "model.")
    classification_layers = get_classification_layers(config, MMENGINE_MODELS, "model.")
    return build_mm_model(config, MMENGINE_MODELS, load_from), classification_layers
