# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Utility functions for mmX build function."""

from __future__ import annotations

import warnings
from copy import deepcopy
from typing import TYPE_CHECKING

from omegaconf import DictConfig

from otx.core.utils.config import convert_conf_to_mmconfig_dict

if TYPE_CHECKING:
    from mmengine.registry import Registry
    from torch import nn


def build_mm_model(config: DictConfig, model_registry: Registry, load_from: str | None = None) -> nn.Module:
    """Build a model by using the registry."""
    from mmengine.logging import MMLogger
    from mmengine.runner import load_checkpoint

    from otx import algo  # noqa: F401

    try:
        model = model_registry.build(convert_conf_to_mmconfig_dict(config, to="tuple"))
    except AssertionError:
        model = model_registry.build(convert_conf_to_mmconfig_dict(config, to="list"))

    mm_logger = MMLogger.get_current_instance()
    mm_logger_level = mm_logger.level
    mm_logger.setLevel("WARNING")
    model.init_weights()
    mm_logger.setLevel(mm_logger_level)
    if load_from is not None:
        load_checkpoint(model, load_from, map_location="cpu")

    return model


def get_default_num_async_infer_requests() -> int:
    """Returns a default number of infer request for OV models."""
    import os

    number_requests = os.cpu_count()
    number_requests = max(1, int(number_requests / 2)) if number_requests is not None else 1
    msg = f"""Set the default number of OpenVINO inference requests to {number_requests}.
            You can specify the value in config."""
    warnings.warn(msg, stacklevel=1)
    return number_requests


def get_classification_layers(
    config: DictConfig,
    model_registry: Registry,
    prefix: str = "",
) -> dict[str, dict[str, int]]:
    """Return classification layer names by comparing two different number of classes models.

    Args:
        config (DictConfig): Config for building model.
        model_registry (Registry): Registry for building model.
        prefix (str): Prefix of model param name.
            Normally it is "model." since OTXModel set it's nn.Module model as self.model

    Return:
        dict[str, dict[str, int]]
        A dictionary contain classification layer's name and information.
        Stride means dimension of each classes, normally stride is 1, but sometimes it can be 4
        if the layer is related bbox regression for object detection.
        Extra classes is default class except class from data.
        Normally it is related with background classes.
    """
    sample_config = deepcopy(config)
    modify_num_classes(sample_config, 5)
    sample_model_dict = build_mm_model(sample_config, model_registry, None).state_dict()

    modify_num_classes(sample_config, 6)
    incremental_model_dict = build_mm_model(sample_config, model_registry, None).state_dict()

    classification_layers = {}
    for key in sample_model_dict:
        if sample_model_dict[key].shape != incremental_model_dict[key].shape:
            sample_model_dim = sample_model_dict[key].shape[0]
            incremental_model_dim = incremental_model_dict[key].shape[0]
            stride = incremental_model_dim - sample_model_dim
            num_extra_classes = 6 * sample_model_dim - 5 * incremental_model_dim
            classification_layers[prefix + key] = {"stride": stride, "num_extra_classes": num_extra_classes}
    return classification_layers


def modify_num_classes(config: DictConfig, num_classes: int) -> None:
    """Modify num_classes of config."""
    for key, value in config.items():
        if key == "num_classes":
            config[key] = num_classes
        elif isinstance(value, (DictConfig, dict)):
            modify_num_classes(value, num_classes)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, (DictConfig, dict)):
                    modify_num_classes(item, num_classes)
