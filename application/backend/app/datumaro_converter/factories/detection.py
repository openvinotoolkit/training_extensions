# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence

import numpy as np
from datumaro.experimental import LazyImage
from datumaro.experimental.fields import ImageInfo, Subset
from loguru import logger

from app.datumaro_converter.domain import DetectionImportExportSample, DetectionTrainingSample
from app.datumaro_converter.utils import ShapeConverter, validate_confidence_consistency
from app.models import DatasetItem, DatasetItemAnnotation, DatasetItemSubset, Media, Rectangle
from app.utils.typing import NDArrayFloat32, NDArrayInt

from .sample_factory import SampleFactory, SampleMode

DetectionSample = DetectionTrainingSample | DetectionImportExportSample


class DetectionSampleFactory(SampleFactory[DetectionSample]):
    """Knows how to create detection samples."""

    _sample_type_map = {
        SampleMode.TRAINING: DetectionTrainingSample,
        SampleMode.IMPORT_EXPORT: DetectionImportExportSample,
    }

    def __create_sample_for_mode(
        self,
        dataset_item: DatasetItem,
        media: Media,
        media_path: str,
        label: NDArrayInt | None = None,
        confidence: NDArrayFloat32 | None = None,
        bboxes: NDArrayInt | None = None,
        user_reviewed: bool = False,
    ) -> DetectionSample | None:
        subset = Subset[dataset_item.subset.name]
        label = np.array([]) if label is None else label
        bboxes = np.array([]) if bboxes is None else bboxes
        match self._mode:
            case SampleMode.IMPORT_EXPORT:
                media_item, media_info = self._get_dm_media_with_info(media, media_path)
                return DetectionImportExportSample(
                    id=str(dataset_item.id),
                    media=media_item,
                    media_info=media_info,
                    subset=subset,
                    label=label,
                    bboxes=bboxes,
                    confidence=confidence,
                    user_reviewed=user_reviewed,
                )
            case SampleMode.TRAINING:
                return DetectionTrainingSample(
                    id=str(dataset_item.id),
                    image=LazyImage(media_path),
                    image_info=ImageInfo(width=media.width, height=media.height),
                    subset=subset,
                    label=label,
                    bboxes=bboxes,
                    confidence=confidence,
                )
            case _:
                raise ValueError(f"Unsupported sample mode: {self._mode}")

    def create_sample(self, dataset_item: DatasetItem, media: Media, media_path: str) -> DetectionSample | None:
        if dataset_item.annotation_data is None:
            return self.__create_sample_for_mode(dataset_item, media, media_path)

        bboxes = self._extract_bboxes(dataset_item.annotation_data)
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
            bboxes=np.array(bboxes),
            user_reviewed=True,
        )

    @staticmethod
    def _extract_bboxes(annotations: Sequence[DatasetItemAnnotation]) -> list[list[int]]:
        return [ShapeConverter.rectangle_to_bbox(ann.shape) for ann in annotations if isinstance(ann.shape, Rectangle)]

    def _extract_label_indices(self, dataset_item: DatasetItem) -> list[int] | None:
        label_ids = [
            ann.labels[0].id
            for ann in dataset_item.annotation_data  # pyrefly: ignore
            if len(ann.labels) == 1
        ]
        indices = self._label_index.get_indices(label_ids)

        if indices is None:
            logger.warning("Label not found for dataset item {}", dataset_item.id)

        return indices

    @staticmethod
    def _extract_confidences(annotations: Sequence[DatasetItemAnnotation]) -> list[float]:
        has_confidences = validate_confidence_consistency(annotations)
        return [ann.confidences[0] for ann in annotations] if has_confidences else []  # pyrefly: ignore
