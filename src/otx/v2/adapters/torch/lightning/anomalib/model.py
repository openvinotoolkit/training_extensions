"""Model build & get list API for OTX anomalib adapter."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING

from anomalib.data import TaskType
from anomalib.models.components import AnomalyModule
from anomalib.post_processing import NormalizationMethod, ThresholdMethod
from anomalib.utils.callbacks import (
    MetricsConfigurationCallback,
    MinMaxNormalizationCallback,
    PostProcessingConfigurationCallback,
)
from omegaconf import DictConfig, OmegaConf

from otx.v2.adapters.torch.lightning.anomalib.registry import AnomalibRegistry
from otx.v2.adapters.torch.lightning.modules.models.base_model import BaseOTXLightningModel
from otx.v2.api.utils.importing import get_files_dict, get_otx_root_path

if TYPE_CHECKING:
    import torch

MODEL_CONFIG_PATH = Path(get_otx_root_path()) / "v2/configs/lightning/models"
MODEL_CONFIGS = get_files_dict(MODEL_CONFIG_PATH)

map_task_type = {str(task_type.name).upper(): task_type for task_type in TaskType}


def get_wrapper_otx_model(model_class: type[AnomalyModule], task: TaskType) -> type[AnomalyModule]:
    """Return a wrapper class for the given model class.

    Args:
        model_class (type[AnomalyModule]): The model class to wrap.
        task (TaskType): The task type of model.

    Returns:
        type: The wrapper class.
    """

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


def get_model(
    model: dict | (DictConfig | str) | None = None,
    checkpoint: str | None = None,
    **kwargs,
) -> torch.nn.Module:
    """Return a torch.nn.Module object based on the provided model configuration or anomalib model api.

    Args:
        model (dict | (DictConfig | str) | None, optional): The model configuration. Can be a dictionary,
            a DictConfig object, or a path to a YAML file containing the configuration.
        checkpoint (str, optional): The path to a checkpoint file to load weights from.
        **kwargs: Additional keyword arguments to pass to the `anomalib_get_model` function.

    Returns:
        torch.nn.Module: The model object.

    """
    kwargs = kwargs or {}
    task = kwargs.pop("task", "classification")
    task = map_task_type[task.upper()]
    if isinstance(model, str):
        if model in MODEL_CONFIGS:
            model = MODEL_CONFIGS[model]
        if Path(model).is_file():
            model = OmegaConf.load(model)
    if not model.get("model", False):
        model = DictConfig(content={"model": model})
    if checkpoint is not None:
        model["init_weights"] = checkpoint
    if isinstance(model, dict):
        model = OmegaConf.create(model)
    if model.model.name.startswith("otx"):
        model.model.name = "_".join(model.model.name.split("_")[1:])
    model_class = AnomalibRegistry().get(model.model.name)
    if isinstance(model_class, type) and issubclass(model_class, AnomalyModule):
        model_class = get_wrapper_otx_model(model_class=model_class, task=task)
        return model_class(model)
    msg = f"Model {model.model.name} is not supported."
    raise NotImplementedError(msg)


def list_models(pattern: str | None = None) -> list[str]:
    """Return a list of available model names.

    Args:
        pattern (str | None, optional): A pattern to filter the model names. Defaults to None.

    Returns:
        list[str]: A sorted list of available model names.
    """
    model_list = list(MODEL_CONFIGS.keys())

    if pattern is not None:
        # Always match keys with any postfix.
        model_list = list(set(fnmatch.filter(model_list, pattern + "*")))

    return sorted(model_list)
