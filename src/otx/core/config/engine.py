# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Config data type objects for engine."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EngineConfig:
    """Configuration class for the engine."""
    max_epochs: int
    precision: int
    val_check_interval: int
    callbacks: list
    accelerator: str
    devices: int

    work_dir: str | None = None
    seed: int | None = None
    deterministic: bool | None = None
    logger: dict | None = None
