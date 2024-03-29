# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Config objects for HPO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch


@dataclass
class HpoConfig:
    """DTO for HPO configuration."""

    search_space: dict[str, dict[str, Any]] | None = None
    save_path: str | None = None
    mode: Literal["max", "min"] = "max"
    num_trials: int | None = None
    num_workers: int = torch.cuda.device_count() if torch.cuda.is_available() else 1
    expected_time_ratio: int | float | None = 4
    maximum_resource: int | float | None = None
    prior_hyper_parameters: dict | list[dict] | None = None
    acceptable_additional_time_ratio: float | int = 1.0
    minimum_resource: int | float | None = None
    reduction_factor: int = 3
    asynchronous_bracket: bool = True
    asynchronous_sha: bool = torch.cuda.device_count() != 1
