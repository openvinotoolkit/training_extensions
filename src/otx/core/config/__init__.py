# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Config data type objects."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from .base import BaseConfig
from .data import DataModuleConfig, InstSegDataModuleConfig
from .model import ModelConfig
from .trainer import TrainerConfig

if TYPE_CHECKING:
    from torch import dtype


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
    resume: bool = False

    seed: Optional[int] = None
    checkpoint: Optional[str] = None


def as_int_tuple(*args) -> tuple[int, ...]:
    """Resolve YAML list into Python integer tuple.

    Example:
        YAML file example::

            ```yaml
            mem_cache_img_max_size: ${as_int_tuple:500,500}
            ```
    """
    return tuple(int(arg) for arg in args)


def as_torch_dtype(arg: str) -> dtype:
    """Resolve YAML string into PyTorch dtype.

    Example:
        YAML file example::

            ```yaml
            transforms:
              - _target_: torchvision.transforms.v2.ToDtype
                dtype: ${as_torch_dtype:torch.float32}
                scale: True
            ```
    """
    import torch

    mapping = {
        "float32": torch.float32,
        "float": torch.float,
        "float64": torch.float64,
        "double": torch.double,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "half": torch.half,
        "uint8": torch.uint8,
        "int8": torch.int8,
        "int16": torch.int16,
        "short": torch.short,
        "int32": torch.int32,
        "int": torch.int,
        "int64": torch.int64,
        "long": torch.long,
        "complex32": torch.complex32,
        "complex64": torch.complex64,
        "cfloat": torch.cfloat,
        "complex128": torch.complex128,
        "cdouble": torch.cdouble,
        "quint8": torch.quint8,
        "qint8": torch.qint8,
        "qint32": torch.qint32,
        "bool": torch.bool,
        "quint4x2": torch.quint4x2,
        "quint2x4": torch.quint2x4,
    }
    prefix = "torch."
    if not arg.startswith(prefix):
        msg = f"arg={arg} should start with the `torch.` prefix"
        raise ValueError(msg)
    key = arg[len(prefix) :]
    return mapping[key]


def register_configs() -> None:
    """Register DTO as default and custom resolvers to hydra."""
    from hydra.core.config_store import ConfigStore
    from omegaconf import OmegaConf

    cs = ConfigStore.instance()
    cs.store(name="base_config", node=TrainConfig)
    cs.store(group="data", name="base_inst_seg", node=InstSegDataModuleConfig)

    OmegaConf.register_new_resolver("as_int_tuple", as_int_tuple, replace=True)
    OmegaConf.register_new_resolver("as_torch_dtype", as_torch_dtype, replace=True)


register_configs()
