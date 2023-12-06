# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Config data type objects."""
from dataclasses import dataclass
from typing import Optional

from .base import BaseConfig
from .data import DataModuleConfig
from .model import ModelConfig
from .trainer import TrainerConfig


@dataclass
class TrainConfig:
    """DTO for training."""

    base: BaseConfig
    callbacks: dict
    data: DataModuleConfig
    trainer: TrainerConfig
    model: ModelConfig
    logger: dict
    recipe: Optional[str]  # noqa: FA100
    debug: Optional[str]  # noqa: FA100
    train: bool
    test: bool


def register_configs() -> None:
    """Register DTO as default to hydra."""
    from hydra.core.config_store import ConfigStore

    cs = ConfigStore.instance()
    cs.store(name="base_config", node=TrainConfig)
