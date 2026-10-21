# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

import numpy as np
from datumaro.experimental import LazyImage
from datumaro.experimental.fields import ImageInfo, Subset
from loguru import logger
from numpy import ndarray

from app.datumaro_converter.domain import InstanceSegmentationImportExportSample, InstanceSegmentationTrainingSample
from app.datumaro_converter.utils import ShapeConverter, validate_confidence_consistency
from app.models import DatasetItem, DatasetItemAnnotation, DatasetItemSubset, Media, Polygon
from app.utils.typing import NDArrayFloat32, NDArrayInt

from .sample_factory import SampleFactory, SampleMode

InstanceSegmentationSample = InstanceSegmentationImportExportSample | InstanceSegmentationTrainingSample


class InstanceSegmentationSampleFactory(SampleFactory[InstanceSegmentationSample]):
    """Knows how to create instance segmentation samples."""

    _sample_type_map = {
        SampleMode.TRAINING: InstanceSegmentationTrainingSample,
        SampleMode.IMPORT_EXPORT: InstanceSegmentationImportExportSample,
    }

    def __create_sample_for_mode(
        self,
        dataset_item: DatasetItem,
        media: Media,
        media_path: str,
        label: NDArrayInt | None = None,
        confidence: NDArrayFloat32 | None = None,
        polygons: NDArrayFloat32 | None = None,
        user_reviewed: bool = False,
    ) -> InstanceSegmentationSample | None:
        label = np.array([]) if label is None else label
        polygons = np.array([]) if polygons is None else polygons
        match self._mode:
            case SampleMode.IMPORT_EXPORT:
                media_item, media_info = self._get_dm_media_with_info(media, media_path)
                return InstanceSegmentationImportExportSample(
                    id=str(dataset_item.id),
                    media=media_item,
                    media_info=media_info,
                    subset=Subset[dataset_item.subset.name],
                    label=label,
                    polygons=polygons,
                    confidence=confidence,
                    user_reviewed=user_reviewed,
                )
            case SampleMode.TRAINING:
                return InstanceSegmentationTrainingSample(
                    id=str(dataset_item.id),
                    image=LazyImage(media_path),
                    image_info=ImageInfo(width=media.width, height=media.height),
                    subset=Subset[dataset_item.subset.name],
                    label=label,
                    polygons=polygons,
                    confidence=confidence,
                )
            case _:
                raise ValueError(f"Unsupported sample mode: {self._mode}")

    def create_sample(
        self, dataset_item: DatasetItem, media: Media, media_path: str
    ) -> InstanceSegmentationSample | None:
        if dataset_item.annotation_data is None:
            return self.__create_sample_for_mode(dataset_item, media, media_path)

        polygons = self._extract_polygons(dataset_item.annotation_data)
        label_indices = self._extract_label_indices(dataset_item)

        if label_indices is None:
            return None
        # Filter out empty-labeled items in the training subset
        if len(label_indices) == 0 and dataset_item.subset == DatasetItemSubset.TRAINING:
            return None

        confidences = self._extract_confidences(dataset_item.annotation_data)

        return self.__create_sample_for_mode(
            dataset_item=dataset_item,
            media=media,
            media_path=media_path,
            label=np.array(label_indices),
            confidence=np.array(confidences) if confidences else None,
            polygons=polygons,
            user_reviewed=True,
        )

    def _extract_polygons(self, annotations: Sequence[DatasetItemAnnotation]) -> ndarray:
        """Extracts polygon coordinates from annotations."""
        polygons = [
            ShapeConverter.polygon_to_points(ann.shape) for ann in annotations if isinstance(ann.shape, Polygon)
        ]
        # Each polygon is a float32 array with dimension (num_vert, 2); since the number of vertices varies
        # between polygons, we need a ragged array (dtype=object) to stack the polygons.
        # While building this array, we must also avoid unwanted type conversions by numpy, hence the use of np.empty
        # and explicit assignment - trying to wrap the polygons simply with np.array([...], dtype=object) would cause
        # the outer array dtype to become float32 instead of object if all polygons have the same number of vertices.
        polygons_np = np.empty(len(polygons), dtype=object)
        polygons_np[:] = [np.asarray(p, dtype=np.float32) for p in polygons]
        return polygons_np

    def _extract_label_indices(self, dataset_item: DatasetItem) -> list[int] | None:
        """Extracts label indices ensuring single label per annotation."""
        label_ids = [
            ann.labels[0].id
            for ann in dataset_item.annotation_data  # pyrefly: ignore
            if len(ann.labels) == 1
        ]

        indices = self._label_index.get_indices(label_ids)

        if indices is None:
            logger.warning("Label not found for dataset item {}", dataset_item.id)

        return indices

    def _extract_confidences(self, annotations: Sequence[DatasetItemAnnotation]) -> list[float]:
        """Extracts confidence scores if consistently present."""
        has_confidences = validate_confidence_consistency(annotations)
        return [ann.confidences[0] for ann in annotations] if has_confidences else []  # pyrefly: ignore
