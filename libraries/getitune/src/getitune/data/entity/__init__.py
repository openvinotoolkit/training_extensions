# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Module for getitune data entities."""

from .base import ImageInfo, ImageType
from .sample import (
    BaseSample,
    Prediction,
    PredictionBatch,
    SampleBatch,
)
from .tile import (
    TileBatchDetDataEntity,
    TileBatchInstSegDataEntity,
    TileBatchSegDataEntity,
    TileDetDataEntity,
    TileSegDataEntity,
)

__all__ = [
    "BaseSample",
    "ImageInfo",
    "ImageType",
    "Prediction",
    "PredictionBatch",
    "SampleBatch",
    "TileBatchDetDataEntity",
    "TileBatchInstSegDataEntity",
    "TileBatchSegDataEntity",
    "TileDetDataEntity",
    "TileSegDataEntity",
]
