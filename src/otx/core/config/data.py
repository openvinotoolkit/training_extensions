# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Config data type objects for data."""
# NOTE: omegaconf would fail to parse dataclass with `from __future__ import annotations` in Python 3.8, 3.9
# ruff: noqa: FA100

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Optional

from otx.core.types.device import DeviceType
from otx.core.types.image import ImageColorChannel
from otx.core.types.transformer_libs import TransformLibType


@dataclass
class SubsetConfig:
    """DTO for dataset subset configuration.

    Attributes:
        batch_size (int): Batch size produced.
        subset_name (str): Datumaro Dataset's subset name for this subset config.
            It can differ from the actual usage (e.g., 'val' for the validation subset config).
        transforms (list[dict[str, Any] | Transform] | Compose): List of actually used transforms.
            It accepts a list of `torchvision.transforms.v2.*` Python objects
            or `torchvision.transforms.v2.Compose` for `TransformLibType.TORCHVISION`.
            Otherwise, it takes a Python dictionary that fits the configuration style used in mmcv
            (`TransformLibType.MMCV`, `TransformLibType.MMPRETRAIN`, ...).
        transform_lib_type (TransformLibType): Transform library type used by this subset.
        num_workers (int): Number of workers for the dataloader of this subset.

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

    # TODO (vinnamki): Revisit data configuration objects to support a union type in structured config
    # Omegaconf does not allow to have a union type, https://github.com/omry/omegaconf/issues/144
    transforms: list[dict[str, Any]]

    transform_lib_type: TransformLibType = TransformLibType.TORCHVISION
    num_workers: int = 2
    sampler: SamplerConfig = field(default_factory=lambda: SamplerConfig())
    to_tv_image: bool = True


@dataclass
class TileConfig:
    """DTO for tiler configuration."""

    enable_tiler: bool = False
    enable_adaptive_tiling: bool = True
    tile_size: tuple[int, int] = (400, 400)
    overlap: float = 0.2
    iou_threshold: float = 0.45
    max_num_instances: int = 1500
    object_tile_ratio: float = 0.03
    sampling_ratio: float = 1.0
    with_full_img: bool = False

    def clone(self) -> TileConfig:
        """Return a deep copied one of this instance."""
        return deepcopy(self)


@dataclass
class VisualPromptingConfig:
    """DTO for visual prompting data module configuration."""

    use_bbox: bool = False
    use_point: bool = False


@dataclass
class DataModuleConfig:
    """DTO for data module configuration."""

    data_format: str
    data_root: str

    train_subset: SubsetConfig
    val_subset: SubsetConfig
    test_subset: SubsetConfig

    tile_config: TileConfig = field(default_factory=lambda: TileConfig())
    vpm_config: VisualPromptingConfig = field(default_factory=lambda: VisualPromptingConfig())

    mem_cache_size: str = "1GB"
    mem_cache_img_max_size: Optional[tuple[int, int]] = None
    image_color_channel: ImageColorChannel = ImageColorChannel.RGB
    stack_images: bool = True

    include_polygons: bool = False
    ignore_index: int = 255
    unannotated_items_ratio: float = 0.0

    auto_num_workers: bool = False
    device: DeviceType = DeviceType.auto


@dataclass
class SamplerConfig:
    """Configuration class for defining the sampler used in the data loading process.

    This is passed in the form of a dataclass, which is instantiated when the dataloader is created.

    [TODO]: Need to replace this with a proper Sampler class.
    Currently, SamplerConfig, which belongs to the sampler of SubsetConfig,
    belongs to the nested dataclass of dataclass, which is not easy to instantiate from the CLI.
    So currently replace sampler with a corresponding dataclass that resembles the configuration of another object,
    providing limited functionality.
    """

    class_path: str = "torch.utils.data.RandomSampler"
    init_args: dict[str, Any] = field(default_factory=dict)
