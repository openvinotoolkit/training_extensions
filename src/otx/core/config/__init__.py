# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Config data type objects."""

from __future__ import annotations

from dataclasses import dataclass

from .base import BaseConfig
from .data import DataModuleConfig
from .model import ModelConfig
from .trainer import TrainerConfig


@dataclass
class TrainConfig:
    """DTO for training.

    Attributes:
        seed: If set it with an integer value, e.g. `seed=1`,
            Lightning derives unique seeds across all dataloader workers and processes
            for torch, numpy and stdlib random number generators.
    """

    base: BaseConfig
    callbacks: dict
    data: DataModuleConfig
    trainer: TrainerConfig
    model: ModelConfig
    logger: dict
    recipe: str | None
    train: bool
    test: bool

    seed: int | None = None


def register_configs() -> None:
    """Register DTO as default to hydra."""
    from hydra.core.config_store import ConfigStore

    cs = ConfigStore.instance()
    cs.store(name="base_config", node=TrainConfig)
