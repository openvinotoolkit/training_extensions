# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Config data type objects for trainer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path  # noqa: TCH003
from typing import Optional

from omegaconf import DictConfig


@dataclass
class TrainerConfig(DictConfig):
    """DTO for trainer configuration."""

    default_root_dir: Path
    accelerator: str
    precision: int
    max_epochs: int
    min_epochs: int
    devices: int
    check_val_every_n_epoch: int
    deterministic: bool
    gradient_clip_val: Optional[float]

    _target_: str = "lightning.pytorch.trainer.Trainer"
