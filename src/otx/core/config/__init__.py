# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Config data type objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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
        checkpoint: The path to the checkpoint file. e.g. `checkpoint=outputs/checkpoints/epoch_000.ckpt`
            Path/URL of the checkpoint from which training is resumed.
            If there is no checkpoint file at the path, an exception is raised.
    """

    base: BaseConfig
    callbacks: dict
    data: DataModuleConfig
    trainer: TrainerConfig
    model: ModelConfig
    logger: dict
    recipe: Optional[str]
    debug: Optional[str]
    train: bool
    test: bool

    seed: Optional[int] = None
    checkpoint: Optional[str] = None


def as_int_tuple(*args) -> tuple[int, ...]:
    """Resolve YAML list into Python integer tuple."""
    return tuple(int(arg) for arg in args)


def register_configs() -> None:
    """Register DTO as default to hydra."""
    from hydra.core.config_store import ConfigStore
    from omegaconf import OmegaConf

    cs = ConfigStore.instance()
    cs.store(name="base_config", node=TrainConfig)

    OmegaConf.register_new_resolver("as_int_tuple", as_int_tuple)
