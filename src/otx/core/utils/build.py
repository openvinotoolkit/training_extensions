# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Utility functions for mmX build function."""

from __future__ import annotations

import warnings
from copy import deepcopy
from typing import TYPE_CHECKING

from mmengine.logging import MMLogger
from omegaconf import DictConfig

from otx.core.utils.config import convert_conf_to_mmconfig_dict

if TYPE_CHECKING:
    from mmengine.registry import Registry
    from torch import nn


def build_mm_model(config: DictConfig, model_registry: Registry, load_from: str | None = None) -> nn.Module:
    """Build a model by using the registry."""
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
        load_checkpoint(model, load_from)

    return model


def get_default_async_reqs_num() -> int:
    """Returns a default number of infer request for OV models."""
    import os

    reqs_num = os.cpu_count()
    reqs_num = max(1, int(reqs_num / 2)) if reqs_num is not None else 1
    msg = f"Set the default number of OpenVINO inference requests to {reqs_num}. You can specify the value in config."
    warnings.warn(msg, stacklevel=1)
    return reqs_num


def get_classification_layers(config: DictConfig, model_registry: Registry, prefix: str = "") -> list[str]:
    """Return classification layer names by comparing two different number of classes models."""
    sample_config = deepcopy(config)
    modify_num_classes(sample_config, 5)
    sample_model_dict = build_mm_model(sample_config, model_registry, None).state_dict()

    modify_num_classes(sample_config, 6)
    incremental_model_dict = build_mm_model(sample_config, model_registry, None).state_dict()

    return [
        prefix + key for key in sample_model_dict if sample_model_dict[key].shape != incremental_model_dict[key].shape
    ]


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
