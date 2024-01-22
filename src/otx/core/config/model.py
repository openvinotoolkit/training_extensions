# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Config data type objects for model."""

from dataclasses import dataclass

# TODO (Eugene): Consider integrating TileConfig into ModelConfig so that OVModel can fold/merge tiles.
# https://github.com/openvinotoolkit/datumaro/pull/1194


@dataclass
class ModelConfig:
    """DTO for model configuration."""

    _target_: str
    optimizer: dict
    scheduler: dict
    otx_model: dict
    torch_compile: bool
