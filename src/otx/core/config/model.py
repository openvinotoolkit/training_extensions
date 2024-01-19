# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Config data type objects for model."""

from dataclasses import dataclass

from src.otx.core.config.explain import ExplainConfig


@dataclass
class ModelConfig:
    """DTO for model configuration."""

    _target_: str
    optimizer: dict
    scheduler: dict
    otx_model: dict
    torch_compile: bool
    explain_config: ExplainConfig
