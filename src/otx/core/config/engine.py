# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Config data type objects for engine."""

from __future__ import annotations

from dataclasses import dataclass

from otx.core.types.task import OTXTaskType


@dataclass
class EngineConfig:
    """Configuration class for the engine."""

    data_root: str
    task: OTXTaskType
    device: str
    work_dir: str = "./otx-workspace"

    checkpoint: str | None = None
