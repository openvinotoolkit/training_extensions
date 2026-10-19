# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import polars as pl
from datumaro.experimental import LazyImage, LazyVideoFrame, MediaInfo, Sample, register_sample
from datumaro.experimental.fields import (
    Subset,
    bbox_field,
    bool_field,
    label_field,
    media_info_field,
    media_path_field,
    numeric_field,
    polygon_field,
    string_field,
    subset_field,
)

from app.models import AnnotationType
from app.utils.typing import NDArrayFloat32, NDArrayInt


class BaseImportExportSample(Sample):
    """
    Base sample class for import/export with shared attributes.

    Attributes:
        id: Unique identifier of the sample
        media: Path to the media (image or video frame)
        media_info: Media information (width, height, etc.)
        subset: Subset name to which the sample belongs to
        user_reviewed: Whether the sample is annotated (True) or unannotated (False)
    """

    id: str | None = string_field(semantic="id")
    media: LazyImage | LazyVideoFrame = media_path_field()
    media_info: MediaInfo = media_info_field()
    subset: Subset = subset_field()
    user_reviewed: bool | None = bool_field(semantic="user_reviewed")


@register_sample
class MulticlassClassificationImportExportSample(BaseImportExportSample):
    """
    Sample for multiclass classification datasets.

    Attributes:
        label: Class label index (0-based)
        confidence: Confidence score for the label. Only for model predictions.
    """

    label: int | None = label_field(dtype=pl.UInt16(), is_list=False)
    confidence: float | None = numeric_field(dtype=pl.Float32(), semantic="confidence")

    @staticmethod
    def annotation_type() -> AnnotationType:
        return AnnotationType.LABEL

    @property
    def annotations(self) -> int:
        return 1 if self.label is not None and self.user_reviewed else 0


@register_sample
class MultilabelClassificationImportExportSample(BaseImportExportSample):
    """
    Sample for multilabel classification datasets.

    Attributes:
        label: Array of class label indices (0-based)
        confidence: Array of confidence scores for each label. Only for model predictions.
    """

    label: NDArrayInt | None = label_field(dtype=pl.UInt16(), multi_label=True)
    confidence: NDArrayFloat32 | None = numeric_field(dtype=pl.Float32(), is_list=True, semantic="confidence")

    @staticmethod
    def annotation_type() -> AnnotationType:
        return AnnotationType.LABEL

    @property
    def annotations(self) -> int:
        is_empty_label = self.user_reviewed and self.label is not None and len(self.label) == 0
        if is_empty_label:
            return 1
        return len(self.label) if self.label is not None and self.user_reviewed else 0


@register_sample
class DetectionImportExportSample(BaseImportExportSample):
    """
    Sample for object detection datasets.

    Attributes:
        bboxes: Array of bounding boxes in x1y1x2y2 format
        label: Array of class label indices (0-based) for each bounding box
        confidence: Array of confidence scores for each bounding box. Only for model predictions.
    """

    bboxes: NDArrayInt | None = bbox_field(dtype=pl.Int32())
    label: NDArrayInt | None = label_field(dtype=pl.UInt16(), is_list=True)
    confidence: NDArrayFloat32 | None = numeric_field(dtype=pl.Float32(), is_list=True, semantic="confidence")

    @staticmethod
    def annotation_type() -> AnnotationType:
        return AnnotationType.BOUNDING_BOX

    @property
    def annotations(self) -> int:
        is_empty_label = self.user_reviewed and self.label is not None and len(self.label) == 0
        if is_empty_label:
            return 1
        return len(self.bboxes) if self.bboxes is not None and self.user_reviewed else 0


@register_sample
class InstanceSegmentationImportExportSample(BaseImportExportSample):
    """
    Sample for instance segmentation datasets.

    Attributes:
        polygons: Array of polygons, each represented as a list of points in xy format
        label: Array of class label indices (0-based) for each polygon
        confidence: Array of confidence scores for each polygon. Only for model predictions.
    """

    polygons: NDArrayFloat32 | None = polygon_field(dtype=pl.Float32())  # Ragged array (num_polygons, num_vertices, 2)
    label: NDArrayInt | None = label_field(dtype=pl.UInt16(), is_list=True)
    confidence: NDArrayFloat32 | None = numeric_field(dtype=pl.Float32(), is_list=True, semantic="confidence")

    @staticmethod
    def annotation_type() -> AnnotationType:
        return AnnotationType.POLYGON

    @property
    def annotations(self) -> int:
        is_empty_label = self.user_reviewed and self.label is not None and len(self.label) == 0
        if is_empty_label:
            return 1
        return len(self.polygons) if self.polygons is not None and self.user_reviewed else 0
