# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence

import numpy as np
from datumaro.experimental import LazyImage
from datumaro.experimental.fields import ImageInfo, Subset
from loguru import logger

from app.datumaro_converter.domain import (
    MultilabelClassificationImportExportSample,
    MultilabelClassificationTrainingSample,
)
from app.datumaro_converter.utils import validate_confidence_consistency
from app.models import DatasetItem, DatasetItemAnnotation, DatasetItemSubset, Media
from app.utils.typing import NDArrayFloat32, NDArrayInt

from .sample_factory import SampleFactory, SampleMode

MultilabelClassificationSample = MultilabelClassificationTrainingSample | MultilabelClassificationImportExportSample


class MultilabelClassificationSampleFactory(SampleFactory[MultilabelClassificationSample]):
    """Knows how to create multilabel classification samples."""

    _sample_type_map = {
        SampleMode.TRAINING: MultilabelClassificationTrainingSample,
        SampleMode.IMPORT_EXPORT: MultilabelClassificationImportExportSample,
    }

    def __create_sample_for_mode(
        self,
        dataset_item: DatasetItem,
        media: Media,
        media_path: str,
        label: NDArrayInt | None = None,
        confidence: NDArrayFloat32 | None = None,
        user_reviewed: bool = False,
    ) -> MultilabelClassificationSample | None:
        match self._mode:
            case SampleMode.IMPORT_EXPORT:
                media_item, media_info = self._get_dm_media_with_info(media, media_path)
                return MultilabelClassificationImportExportSample(
                    id=str(dataset_item.id),
                    media=media_item,
                    media_info=media_info,
                    subset=Subset[dataset_item.subset.name],
                    label=np.array([]) if label is None else label,
                    confidence=confidence,
                    user_reviewed=user_reviewed,
                )
            case SampleMode.TRAINING:
                return MultilabelClassificationTrainingSample(
                    id=str(dataset_item.id),
                    image=LazyImage(media_path),
                    image_info=ImageInfo(width=media.width, height=media.height),
                    subset=Subset[dataset_item.subset.name],
                    label=np.array([]) if label is None else label,
                    confidence=confidence,
                )
            case _:
                raise ValueError(f"Unsupported sample mode: {self._mode}")

    def create_sample(
        self, dataset_item: DatasetItem, media: Media, media_path: str
    ) -> MultilabelClassificationSample | None:
        if dataset_item.annotation_data is None:
            return self.__create_sample_for_mode(dataset_item, media, media_path)

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
            user_reviewed=True,
        )

    def _extract_label_indices(self, dataset_item: DatasetItem) -> list[int] | None:
        """Extracts indices for all labels in the annotations."""
        label_ids = [label.id for ann in dataset_item.annotation_data for label in ann.labels]  # pyrefly: ignore

        indices = self._label_index.get_indices(label_ids)

        if indices is None:
            logger.warning("Label not found for dataset item {}", dataset_item.id)

        return indices

    @staticmethod
    def _extract_confidences(annotations: Sequence[DatasetItemAnnotation]) -> list[float]:
        """Extracts confidence scores if consistently present."""
        has_confidences = validate_confidence_consistency(annotations)

        if not has_confidences:
            return []

        confidences = []
        for ann in annotations:
            if ann.confidences:
                confidences.extend(ann.confidences)

        return confidences
