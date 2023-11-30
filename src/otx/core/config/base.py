# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Config data type objects for base."""

from dataclasses import dataclass
from pathlib import Path

from otx.core.types.task import OTXTaskType


@dataclass
class BaseConfig:
    """DTO for base configuration."""
    task: OTXTaskType

    work_dir: Path
    data_dir: Path
    log_dir: Path
    output_dir: Path
