# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import polars as pl
from datumaro.experimental import LazyImage, Sample, register_sample
from datumaro.experimental.fields import (
    ImageInfo,
    Subset,
    bbox_field,
    image_info_field,
    image_path_field,
    label_field,
    numeric_field,
    polygon_field,
    string_field,
    subset_field,
)

from app.utils.typing import NDArrayFloat32, NDArrayInt


class BaseTrainingSample(Sample):
    """
    Base sample class for training with shared attributes.

    Attributes:
        id: Unique identifier of the sample
        image: Path to the image
        image_info: Image information (width, height)
        subset: Subset name to which the sample belongs to
    """

    id: str = string_field(semantic="id")
    image: LazyImage = image_path_field()
    image_info: ImageInfo = image_info_field()
    subset: Subset = subset_field()


@register_sample
class MulticlassClassificationTrainingSample(BaseTrainingSample):
    """
    Sample for multiclass classification datasets.

    Attributes:
        label: Class label index (0-based)
        confidence: Confidence score for the label. Only for model predictions.
    """

    label: int | None = label_field(dtype=pl.UInt16(), is_list=False)
    confidence: float | None = numeric_field(dtype=pl.Float32(), semantic="confidence")


@register_sample
class MultilabelClassificationTrainingSample(BaseTrainingSample):
    """
    Sample for multilabel classification datasets.

    Attributes:
        label: Array of class label indices (0-based)
        confidence: Array of confidence scores for each label. Only for model predictions.
    """

    label: NDArrayInt = label_field(dtype=pl.UInt16(), multi_label=True)
    confidence: NDArrayFloat32 | None = numeric_field(dtype=pl.Float32(), is_list=True, semantic="confidence")


@register_sample
class DetectionTrainingSample(BaseTrainingSample):
    """
    Sample for object detection datasets.

    Attributes:
        bboxes: Array of bounding boxes in x1y1x2y2 format
        label: Array of class label indices (0-based) for each bounding box
        confidence: Array of confidence scores for each bounding box. Only for model predictions.
    """

    bboxes: NDArrayInt = bbox_field(dtype=pl.Int32())
    label: NDArrayInt = label_field(dtype=pl.UInt16(), is_list=True)
    confidence: NDArrayFloat32 | None = numeric_field(dtype=pl.Float32(), is_list=True, semantic="confidence")


@register_sample
class InstanceSegmentationTrainingSample(BaseTrainingSample):
    """
    Sample for instance segmentation datasets.

    Attributes:
        polygons: Array of polygons, each represented as a list of points in xy format
        label: Array of class label indices (0-based) for each polygon
        confidence: Array of confidence scores for each polygon. Only for model predictions.
    """

    polygons: NDArrayFloat32 = polygon_field(dtype=pl.Float32())  # Ragged array (num_polygons, num_vertices, 2)
    label: NDArrayInt = label_field(dtype=pl.UInt16(), is_list=True)
    confidence: NDArrayFloat32 | None = numeric_field(dtype=pl.Float32(), is_list=True, semantic="confidence")
