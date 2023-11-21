"""Wrapper BaseOTXLightningModel for anomalib Module."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from anomalib.data import TaskType
from anomalib.post_processing import NormalizationMethod, ThresholdMethod
from anomalib.utils.callbacks import (
    MetricsConfigurationCallback,
    MinMaxNormalizationCallback,
    PostProcessingConfigurationCallback,
)

from otx.v2.adapters.torch.lightning.modules.models.base_model import BaseOTXLightningModel

if TYPE_CHECKING:
    from anomalib.models.components import AnomalyModule
    from omegaconf import DictConfig

map_task_type = {str(task_type.name).upper(): task_type for task_type in TaskType}


def get_wrapper_otx_model(model_class: type[AnomalyModule], task: str | TaskType) -> type[AnomalyModule]:
    """Return a wrapper class for the given model class.

    Args:
        model_class (type[AnomalyModule]): The model class to wrap.
        task (str | TaskType): The task type of model.

    Returns:
        type: The wrapper class.
    """
    if isinstance(task, str):
        task = map_task_type[task.upper()]

    class OTXAnomalibModel(BaseOTXLightningModel, model_class):
        def __init__(self, hparams: DictConfig) -> None:
            super().__init__(hparams)
            self.config = hparams

        @property
        def callbacks(self) -> list:
            metrics = self.config.pop("metrics", {})
            metric_threshold = metrics.get("threshold", {})
            return [
                MinMaxNormalizationCallback(),
                MetricsConfigurationCallback(
                    task=task,
                    image_metrics=metrics.get("image", None),
                    pixel_metrics=metrics.get("pixel", None),
                ),
                PostProcessingConfigurationCallback(
                    normalization_method=NormalizationMethod.MIN_MAX,
                    threshold_method=ThresholdMethod.ADAPTIVE,
                    manual_image_threshold=metric_threshold.get("manual_image", None),
                    manual_pixel_threshold=metric_threshold.get("manual_pixel", None),
                ),
            ]

    return OTXAnomalibModel
