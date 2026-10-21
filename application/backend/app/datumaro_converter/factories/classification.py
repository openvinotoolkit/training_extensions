# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from datumaro.experimental import LazyImage
from datumaro.experimental.fields import ImageInfo, Subset
from loguru import logger

from app.datumaro_converter.domain import (
    MulticlassClassificationImportExportSample,
    MulticlassClassificationTrainingSample,
)
from app.models import DatasetItem, Media

from .sample_factory import SampleFactory, SampleMode

MulticlassClassificationSample = MulticlassClassificationImportExportSample | MulticlassClassificationTrainingSample


class ClassificationSampleFactory(SampleFactory[MulticlassClassificationSample]):
    """Knows how to create classification samples."""

    _sample_type_map = {
        SampleMode.TRAINING: MulticlassClassificationTrainingSample,
        SampleMode.IMPORT_EXPORT: MulticlassClassificationImportExportSample,
    }

    def __create_sample_for_mode(
        self,
        dataset_item: DatasetItem,
        media: Media,
        media_path: str,
        label: int | None = None,
        confidence: float | None = None,
        user_reviewed: bool = False,
    ) -> MulticlassClassificationSample | None:
        subset = Subset[dataset_item.subset.name]
        match self._mode:
            case SampleMode.IMPORT_EXPORT:
                media_item, media_info = self._get_dm_media_with_info(media, media_path)
                return MulticlassClassificationImportExportSample(
                    id=str(dataset_item.id),
                    media=media_item,
                    media_info=media_info,
                    subset=subset,
                    label=label,
                    confidence=confidence,
                    user_reviewed=user_reviewed,
                )
            case SampleMode.TRAINING:
                return MulticlassClassificationTrainingSample(
                    id=str(dataset_item.id),
                    image=LazyImage(media_path),
                    image_info=ImageInfo(width=media.width, height=media.height),
                    subset=subset,
                    label=label,
                    confidence=confidence,
                )
            case _:
                raise ValueError(f"Unsupported sample mode: {self._mode}")

    def create_sample(
        self, dataset_item: DatasetItem, media: Media, media_path: str
    ) -> MulticlassClassificationSample | None:
        if dataset_item.annotation_data is None:
            return self.__create_sample_for_mode(dataset_item, media, media_path)

        if len(dataset_item.annotation_data) == 0:
            raise ValueError(
                f"Expected exactly one annotation for classification project, found empty list. "
                f"Project ID {dataset_item.project_id}, Dataset Item ID: {dataset_item.id}"
            )
        annotation = dataset_item.annotation_data[0]
        label_index = self._label_index.get_index(annotation.labels[0].id)

        if label_index is None:
            logger.warning("Label not found for dataset item {}", dataset_item.id)
            return None

        return self.__create_sample_for_mode(
            dataset_item=dataset_item,
            media=media,
            media_path=media_path,
            label=label_index,
            confidence=annotation.confidences[0] if annotation.confidences else None,
            user_reviewed=True,
        )
