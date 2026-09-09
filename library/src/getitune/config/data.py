# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Config data type objects for data."""
# NOTE: omegaconf would fail to parse dataclass with `from __future__ import annotations` in Python 3.8, 3.9

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


@dataclass
class IntensityConfig:
    """Configuration for intensity mapping (uint8, uint16, or other high-bit-depth inputs).

    Controls how raw pixel values are converted to float32 [0, 1] before augmentations.
    For standard uint8 images the default ``mode="scale_to_unit"`` divides by 255.
    For high-bit-depth inputs (uint16 thermal, medical, etc.) select an appropriate mode.

    Supported modes:
        - ``"scale_to_unit"``: Divide by ``max_value`` and clamp to [0, 1].
          Default for both uint8 (max_value=255) and uint16 (max_value=65535).
        - ``"window"``: Window/level mapping for CT-style imaging.
          Clips to ``[window_center - window_width/2, window_center + window_width/2]``
          then normalizes to [0, 1].
        - ``"percentile"``: Per-image percentile clipping for microscopy / variable-range data.
          Clips to ``[percentile_low, percentile_high]`` quantiles then normalizes.
        - ``"range_scale"``: Multiply by ``scale_factor``, clip to ``[min_value, max_value]``,
          normalize to [0, 1].  Designed for thermal cameras where raw pixel values
          need a conversion factor and a physical temperature range (see ``process_raw_thermal.py``).

    Attributes:
        storage_dtype: Input storage dtype: ``"uint8"`` | ``"uint16"`` | ``"int16"`` | ``"float32"``.
            Determines the Polars/Datumaro schema used for image decode.
        max_value: Maximum raw value for ``"scale_to_unit"`` mode.
            ``None`` = auto (255 for uint8, 65535 for uint16, 32767 for int16).
        mode: Intensity mapping mode (see above).
        window_center: Center of the intensity window (``"window"`` mode).
        window_width: Width of the intensity window (``"window"`` mode).
        percentile_low: Low percentile for clipping (``"percentile"`` mode, default 1.0).
        percentile_high: High percentile for clipping (``"percentile"`` mode, default 99.0).
        scale_factor: Multiplicative factor applied to raw pixels before clipping
            (``"range_scale"`` mode, e.g. 0.4 for thermal Kelvin conversion).
        min_value: Minimum physical value after scaling, used as clip lower bound
            (``"range_scale"`` mode, e.g. 295.15 K).
        repeat_channels: If > 0, repeat single-channel images to this many channels
            (e.g. 3 for pretrained RGB backbones). 0 = no repeat.
    """

    storage_dtype: str = "uint8"
    max_value: float | None = None
    mode: str = "scale_to_unit"
    window_center: float | None = None
    window_width: float | None = None
    percentile_low: float = 1.0
    percentile_high: float = 99.0
    scale_factor: float = 1.0
    min_value: float = 0.0
    repeat_channels: int = 0


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


@dataclass
class SubsetConfig:
    """DTO for dataset subset configuration.

    Attributes:
        batch_size (int): Batch size produced by the dataloader.
        subset_name (str): Datumaro Dataset's subset name for this subset config.
        augmentations_cpu (list[dict[str, Any]]): CPU-stage augmentations using torchvision.transforms.v2.
            These run in Dataset workers before collate. Must output fixed-size tensors for batching.
            Examples: Resize, RandomResizedCrop, intensity mapping transforms.
        augmentations_gpu (list[dict[str, Any]]): GPU-stage augmentations using Kornia.
            These run after batch transfer to GPU via Lightning Callback.
            Examples: RandomHorizontalFlip, ColorJiggle, Normalize.
        intensity (IntensityConfig): High-bit-depth intensity mapping configuration.
        num_workers (int): Number of worker processes for the dataloader.
        sampler (SamplerConfig): Sampler configuration for the dataloader.
        input_size (tuple[int, int] | None): Input size expected by the model.
            If `$(input_size)` is present in augmentations, it will be replaced with this value.

    Example:
        ```python
        train_subset_config = SubsetConfig(
            batch_size=64,
            subset_name="train",
            augmentations_cpu=[
                {"class_path": "torchvision.transforms.v2.RandomResizedCrop", "init_args": {"size": (224, 224)}},
            ],
            augmentations_gpu=[
                {"class_path": "kornia.augmentation.RandomHorizontalFlip", "init_args": {"p": 0.5}},
                {"class_path": "kornia.augmentation.Normalize", "init_args": {"mean": [0.485, 0.456, 0.406],
                                                                              "std": [0.229, 0.224, 0.225]}},
            ],
            num_workers=2,
        )
        ```
    """

    batch_size: int = 6
    subset_name: str = "train"
    augmentations_cpu: list[dict[str, Any]] = field(default_factory=list)
    augmentations_gpu: list[dict[str, Any]] = field(default_factory=list)
    intensity: IntensityConfig = field(default_factory=IntensityConfig)
    num_workers: int = 2
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    input_size: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        """Normalize nested config objects loaded from recipe dictionaries."""
        if isinstance(self.intensity, dict):
            self.intensity = IntensityConfig(**self.intensity)
        if isinstance(self.sampler, dict):
            self.sampler = SamplerConfig(**self.sampler)


@dataclass
class TileConfig:
    """DTO for tiler configuration.

    Attributes:
        tile_inference_batch_size: Number of tiles forwarded through the model at once
            during tiled inference. This is decoupled from the dataloader batch size
            (the number of original images per batch): a single image expands into many
            tiles, and this value controls how many of those tiles are grouped into a
            single model forward pass for GPU efficiency. Configurable from the recipe.
    """

    enable_tiler: bool = False
    enable_adaptive_tiling: bool = True
    tile_size: tuple[int, int] = (400, 400)
    overlap: float = 0.2
    iou_threshold: float = 0.45
    max_num_instances: int = 1500
    object_tile_ratio: float = 0.03
    sampling_ratio: float = 1.0
    tile_inference_batch_size: int = 8

    def __repr__(self):
        """Return a string representation of the TileConfig instance."""
        return (
            f"TileConfig(enable_tiler={self.enable_tiler}, "
            f"enable_adaptive_tiling={self.enable_adaptive_tiling}, "
            f"tile_size={self.tile_size}, "
            f"overlap={self.overlap}, "
            f"iou_threshold={self.iou_threshold}, "
            f"max_num_instances={self.max_num_instances}, "
            f"object_tile_ratio={self.object_tile_ratio}, "
            f"sampling_ratio={self.sampling_ratio}, "
            f"tile_inference_batch_size={self.tile_inference_batch_size})"
        )

    def clone(self) -> TileConfig:
        """Return a deep copied one of this instance."""
        return deepcopy(self)
