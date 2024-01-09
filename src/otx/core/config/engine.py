# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Config data type objects for engine."""

from __future__ import annotations

from dataclasses import dataclass

from otx.core.types.task import OTXTaskType


@dataclass
class EngineConfig:
    """Configuration class for the engine."""

    task: OTXTaskType
    device: str
    work_dir: str = "./otx-workspace"

    data_root: str | None = None
    checkpoint: str | None = None
