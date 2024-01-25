# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Config data type objects for data."""
# NOTE: omegaconf would fail to parse dataclass with `from __future__ import annotations` in Python 3.8, 3.9
# ruff: noqa: FA100

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from otx.core.types.image import ImageColorChannel
from otx.core.types.transformer_libs import TransformLibType

if TYPE_CHECKING:
    from torchvision.transforms.v2 import Compose, Transform


@dataclass
class SubsetConfig:
    """DTO for dataset subset configuration.

    Attributes:
        batch_size: Batch size produced.
        subset_name: Datumaro Dataset's subset name for this subset config.
            It can differ from the actual usage (e.g., 'val' for the validation subset config).
        transforms: List of actually used transforms.
            It accepts a list of `torchvision.transforms.v2.*` Python objects
            or `torchvision.transforms.v2.Compose` for `TransformLibType.TORCHVISION`.
            Otherwise, it takes a Python dictionary that fits the configuration style used in mmcv
            (`TransformLibType.MMCV`, `TransformLibType.MMPRETRAIN`, ...).
        transform_lib_type: Transform library type used by this subset.
        num_workers: Number of workers for the dataloader of this subset.

    Example:
        ```python
        train_subset_config = SubsetConfig(
            batch_size=64,
            subset_name="train",
            transforms=v2.Compose(
                [
                    v2.RandomResizedCrop(size=(224, 224), antialias=True),
                    v2.RandomHorizontalFlip(p=0.5),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ],
            )
            transform_lib_type=TransformLibType.TORCHVISION,
            num_workers=2,
        )
        ```
    """

    batch_size: int
    subset_name: str

    transforms: list[dict[str, Any] | Transform] | Compose

    transform_lib_type: TransformLibType = TransformLibType.TORCHVISION
    num_workers: int = 2


@dataclass
class TilerConfig:
    """DTO for tiler configuration."""

    enable_tiler: bool = False
    grid_size: tuple[int, int] = (2, 2)
    overlap: float = 0.0


@dataclass
class DataModuleConfig:
    """DTO for data module configuration."""

    data_format: str
    data_root: str

    train_subset: SubsetConfig
    val_subset: SubsetConfig
    test_subset: SubsetConfig

    tile_config: TilerConfig

    mem_cache_size: str = "1GB"
    mem_cache_img_max_size: Optional[tuple[int, int]] = None
    image_color_channel: ImageColorChannel = ImageColorChannel.RGB

    include_polygons: bool = False
