# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Config objects for HPO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class HpoConfig:
    """DTO for HPO configuration."""

    search_space: dict[str, dict[str, Any]] | None = None
    save_path: str | None = None
    mode: Literal["max", "min"] = "max"
    num_trials: int | None = None
    num_workers: int | None = None
    num_full_iterations: int | float | None = None
    non_pure_train_ratio: float | None = None
    full_dataset_size: int | None = None
    expected_time_ratio: int | float | None = 4
    maximum_resource: int | float | None = None
    subset_ratio: float | int | None = None
    min_subset_size: int | None = None
    prior_hyper_parameters: dict | list[dict] | None = None
    acceptable_additional_time_ratio: float | int = 1.0
    minimum_resource: int | float | None = None
    reduction_factor: int | None = None
    asynchronous_sha: bool = True
    asynchronous_bracket: bool = False
