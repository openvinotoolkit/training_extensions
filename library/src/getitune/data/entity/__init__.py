# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Module for getitune data entities."""

from .base import ImageInfo, ImageType
from .polygons import (
    clip_polygons,
    concat_polygons,
    empty_polygons,
    pad_polygons,
    polygons_from_object_array,
    scale_polygons,
)
from .rasterization import rasterize, rasterize_crop
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
    "clip_polygons",
    "concat_polygons",
    "empty_polygons",
    "pad_polygons",
    "polygons_from_object_array",
    "rasterize",
    "rasterize_crop",
    "scale_polygons",
]
