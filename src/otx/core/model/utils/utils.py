# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Utility files."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from omegaconf import DictConfig


def get_mean_std_from_data_processing(config: DictConfig) -> dict[str, Any]:
    return {
        "mean": config["data_preprocessor"]["mean"],
        "std": config["data_preprocessor"]["std"],
    }
