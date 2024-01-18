# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Config data type objects for data."""
# NOTE: omegaconf would fail to parse dataclass with `from __future__ import annotations` in Python 3.8, 3.9
# ruff: noqa: FA100
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from otx.core.types.image import ImageColorChannel
from otx.core.types.transformer_libs import TransformLibType


@dataclass
class SubsetConfig:
    """DTO for dataset subset configuration."""

    batch_size: int
    subset_name: str

    transform_lib_type: TransformLibType
    transforms: List[Dict[str, Any]]

    num_workers: int = 2


@dataclass
class DataModuleConfig:
    """DTO for data module configuration."""

    data_format: str
    data_root: str

    train_subset: SubsetConfig
    val_subset: SubsetConfig
    test_subset: SubsetConfig

    mem_cache_size: str = "1GB"
    mem_cache_img_max_size: Optional[Tuple[int, int]] = None
    image_color_channel: ImageColorChannel = ImageColorChannel.RGB
    stack_images: bool = True


@dataclass
class InstSegDataModuleConfig(DataModuleConfig):
    """DTO for instance segmentation data module configuration."""

    include_polygons: bool = True
