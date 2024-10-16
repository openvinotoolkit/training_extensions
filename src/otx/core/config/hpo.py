# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Config objects for HPO."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path  # noqa: TCH003
from typing import Any, Callable, Literal

import torch

from otx.utils.utils import is_xpu_available

if torch.cuda.is_available():
    num_workers = torch.cuda.device_count()
elif is_xpu_available():
    num_workers = torch.xpu.device_count()
else:
    num_workers = 1


@dataclass
class HpoConfig:
    """DTO for HPO configuration.

    progress_update_callback (Callable[[int | float], None] | None):
        callback to update progress. If it's given, it's called with progress every second.
    callbacks_to_exclude (list[str] | str | None): List of name of callbacks to exclude during HPO.
    """

    search_space: dict[str, dict[str, Any]] | str | Path | None = None
    save_path: str | None = None
    mode: Literal["max", "min"] = "max"
    num_trials: int | None = None
    num_workers: int = num_workers
    expected_time_ratio: int | float | None = 4
    maximum_resource: int | float | None = None
    prior_hyper_parameters: dict | list[dict] | None = None
    acceptable_additional_time_ratio: float | int = 1.0
    minimum_resource: int | float | None = None
    reduction_factor: int = 3
    asynchronous_bracket: bool = True
    asynchronous_sha: bool = num_workers > 1
    metric_name: str | None = None
    adapt_bs_search_space_max_val: Literal["None", "Safe", "Full"] = "None"
    progress_update_callback: Callable[[int | float], None] | None = None
    callbacks_to_exclude: list[str] | str | None = None
