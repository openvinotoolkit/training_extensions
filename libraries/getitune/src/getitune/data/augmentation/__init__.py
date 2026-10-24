# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""getitune Augmentation module.

This module provides a two-stage augmentation pipeline:
- CPU stage: Size-dependent augmentations in Dataset workers (torchvision.transforms.v2)
- GPU stage: Batch-level augmentations via Lightning Callback (Kornia)
- Intensity mapping: High-bit-depth (uint16) → float32 [0,1] conversion
"""

# Import kernels module to ensure ImageInfo torchvision kernel registrations are active.
from getitune.data.augmentation.intensity import (
    PercentileClip,
    RangeScale,
    RepeatChannels,
    ScaleToUnit,
    WindowLevel,
    build_intensity_transform,
)
from getitune.data.augmentation.pipeline import (
    CPUAugmentationPipeline,
    GPUAugmentationPipeline,
)

from . import kernels

__all__ = [
    "CPUAugmentationPipeline",
    "GPUAugmentationPipeline",
    "PercentileClip",
    "RangeScale",
    "RepeatChannels",
    "ScaleToUnit",
    "WindowLevel",
    "build_intensity_transform",
    "kernels",
]
