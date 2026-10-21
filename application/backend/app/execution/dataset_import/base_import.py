# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import cast
from uuid import UUID

import numpy as np
from datumaro.experimental import Dataset, LazyImage, LazyVideoFrame, Sample
from datumaro.experimental.categories import LabelCategories
from loguru import logger
from sqlalchemy.orm import Session

from app.datumaro_converter import (
    DetectionImportExportSample,
    InstanceSegmentationImportExportSample,
    MulticlassClassificationImportExportSample,
    MultilabelClassificationImportExportSample,
    import_dataset,
)
from app.execution.base import Execution, JobParamsT
from app.models import DatasetItemSubset, Media, MediaType, Task, TaskType
from app.models.media import ImageFormat, VideoFormat
from app.services import DatasetService, LabelService, MediaService
from app.services.media_service import ImageMetadata

from .sample_to_annotation import DatumaroSampleToGetiAnnotationConverter


class BaseDatasetImport(Execution[JobParamsT], ABC):
    """
    Base implementation for dataset import logic, inheriting from Execution.

    This class provides protected helper methods (_import_dataset, _convert_dataset, _create_items)
    that can be orchestrated by concrete subclasses using the @step decorator.
    It does not define the @step orchestration itself.

    **Note**: Items are currently created one by one in sequential order. This approach may impact
    performance for large datasets and could be optimized by implementing batch processing to reduce
    database overhead and improve throughput.

    Progress is reported in increments of 5% during the import process.

    Attributes:
        BATCH_PROGRESS_INTERVAL: Number of batches for progress reporting (20 batches = 5% intervals).

    Args:
        staged_datasets_dir: Path to the directory containing staged dataset files.
        dataset_service: Service for managing dataset items and operations.
        label_service: Service for managing project labels.
        media_service: Service for managing media items (images).
        db_session_factory: Factory for creating database sessions.

    Raises:
        ValueError: If the staged dataset directory does not exist or dataset import fails.
    """

    BATCH_PROGRESS_INTERVAL = 20  # 5% intervals (100% / 5% = 20)

    SUPPORTED_CONVERSIONS: dict[type[Sample], list[type[Sample]]] = {
        DetectionImportExportSample: [
            InstanceSegmentationImportExportSample,
            MultilabelClassificationImportExportSample,
        ],
        InstanceSegmentationImportExportSample: [
            DetectionImportExportSample,
            MultilabelClassificationImportExportSample,
        ],
        MultilabelClassificationImportExportSample: [],
        MulticlassClassificationImportExportSample: [],
    }

    def __init__(
        self,
        staged_datasets_dir: Path,
        dataset_service: DatasetService,
        label_service: LabelService,
        media_service: MediaService,
        db_session_factory: Callable[[], AbstractContextManager[Session]],
    ) -> None:
        super().__init__()
        self._staged_datasets_dir = staged_datasets_dir
        self._media_service = media_service
        self._label_service = label_service
        self._dataset_service = dataset_service
        self._db_session_factory = db_session_factory

    def _import_dataset(self, staged_dataset_id: UUID) -> Dataset:
        staged_dataset_path = self._staged_datasets_dir / str(staged_dataset_id) / "dataset"
        if not staged_dataset_path.exists() or not staged_dataset_path.is_dir():
            raise ValueError(f"Staged dataset directory does not exist: {staged_dataset_path}")
        return import_dataset(str(staged_dataset_path))

    def _convert_dataset(self, dataset: Dataset, task: Task) -> Dataset:
        dataset_type = dataset.dtype
        target_type = self.__get_sample_by_task(task=task)
        if target_type != dataset_type:
            if (
                dataset_type in self.SUPPORTED_CONVERSIONS
                and target_type not in self.SUPPORTED_CONVERSIONS[dataset_type]
            ):
                raise ValueError(
                    f"Dataset type {dataset_type.__name__} conversion to {target_type.__name__} is not supported. "
                    f"Supported conversions are {list(self.SUPPORTED_CONVERSIONS[dataset_type])}"
                )
            dataset = dataset.convert_to_schema(target_type)
        return dataset

    def _create_items(  # noqa: C901
        self,
        dataset: Dataset,
        project_id: UUID,
        task: Task,
        labels_mapping: dict[str, str | None],
        include_unannotated: bool,
        start_progress: float = 10.0,
    ) -> None:
        with self._db_session_factory() as session:
            self._dataset_service.set_db_session(session)
            self._label_service.set_db_session(session)
            self._media_service.set_db_session(session)
            project_labels = self._label_service.list_all(project_id)
            dataset_label_cats = self._get_dataset_label_categories(dataset)
            converter = DatumaroSampleToGetiAnnotationConverter(
                project_labels=project_labels,
                label_categories=dataset_label_cats,
                label_mapping=labels_mapping,
            )
            logger.info("Found {} labels for project {}", [label.name for label in project_labels], project_id)
            unfiltered_dataset_size, min_p, max_p = len(dataset), start_progress, 100
            progress_interval = max(1, unfiltered_dataset_size // self.BATCH_PROGRESS_INTERVAL)
            num_imported_images, num_imported_frames = 0, 0
            # Cache to track created videos and their IDs to avoid duplicates when importing video frames
            created_videos: dict[str, UUID] = {}
            for idx, item in enumerate(dataset):
                annotations, user_reviewed = [], item.user_reviewed
                # apply conversion only if it's not an Empty label case
                empty_label = (
                    user_reviewed is True
                    and isinstance(item.label, np.ndarray)
                    and item.label.ndim == 1
                    and item.label.size == 0
                )
                if not empty_label:
                    annotations = converter.convert_sample(item) or None
                    # non-native *ImportExportSample types are always treated as reviewed
                    user_reviewed = user_reviewed if user_reviewed is not None else True
                    # If there are no annotations (due to filtering), we consider the item as not reviewed.
                    if not annotations:
                        user_reviewed = False
                    if not user_reviewed and not include_unannotated:
                        continue

                name_suffix = str(idx).zfill(len(str(unfiltered_dataset_size)))
                media: Media | None = None
                match item.media:
                    case LazyImage():
                        media = self._media_service.create_image(
                            ImageMetadata(
                                project_id=project_id,
                                name=f"image_{name_suffix}",
                                image_format=self.__detect_image_format(item.media),
                                data=item.media.data,
                            )
                        )
                        num_imported_images += 1
                    case LazyVideoFrame(video_path=video_path, frame_index=frame_idx):
                        video_path = str(video_path)
                        if video_path not in created_videos:
                            with Path(video_path).open("rb") as video_data:
                                video = self._media_service.create_video(
                                    project_id=project_id,
                                    name=f"video_{name_suffix}",
                                    video_format=self.__detect_video_format(item.media),
                                    data=video_data,
                                )
                                created_videos[video_path] = video.id
                        media = self._media_service.create_image(
                            ImageMetadata(
                                project_id=project_id,
                                name=f"video_frame_{name_suffix}",
                                image_format=ImageFormat.JPG,
                                media_type=MediaType.VIDEO_FRAME,
                                data=item.media.data,
                                video_id=created_videos[video_path],
                                frame_idx=frame_idx,
                            )
                        )
                        num_imported_frames += 1

                if media is None:
                    raise ValueError(f"Error creating media for item {item}")

                self._dataset_service.create_dataset_item(
                    project_id=project_id,
                    task=task,
                    media=media,
                    user_reviewed=user_reviewed,
                    annotations=annotations,
                    subset=DatasetItemSubset(item.subset.name.lower()),
                )
                if (idx > 0 and idx % progress_interval == 0) or idx == unfiltered_dataset_size - 1:
                    self.update_progress(min_p + ((idx + 1) / unfiltered_dataset_size) * (max_p - min_p))
            num_imported_media = num_imported_images + num_imported_frames
            if num_imported_media == 0:
                self.pin_message(
                    "No items were imported from the dataset. "
                    "This may be due to filtering options that excluded all items."
                )
            else:
                self.pin_message(
                    f"Imported {num_imported_media}/{unfiltered_dataset_size} items "
                    f"({num_imported_images} image(s), {len(created_videos)} video(s), "
                    f"{num_imported_frames} frame(s)).",
                    level="INFO",
                )

    @staticmethod
    def _get_dataset_label_categories(dataset: Dataset) -> LabelCategories:
        label_attr_name = "label" if "label" in dataset.schema.attributes else "labels"
        label_attr = dataset.schema.attributes[label_attr_name]
        return cast(LabelCategories, label_attr.categories)

    @staticmethod
    def __detect_image_format(image: LazyImage) -> ImageFormat:
        try:
            ext = Path(image.path).suffix.lstrip(".").lower()
            return ImageFormat(ext)
        except (ValueError, AttributeError):
            return ImageFormat.JPG

    @staticmethod
    def __detect_video_format(video_frame: LazyVideoFrame) -> VideoFormat:
        try:
            ext = Path(video_frame.video_path).suffix.lstrip(".").lower()
            return VideoFormat(ext)
        except (ValueError, AttributeError):
            return VideoFormat.MP4

    @staticmethod
    def __get_sample_by_task(task: Task) -> type[Sample]:
        match task.task_type:
            case TaskType.CLASSIFICATION:
                if not task.exclusive_labels:
                    return MultilabelClassificationImportExportSample
                return MulticlassClassificationImportExportSample
            case TaskType.DETECTION:
                return DetectionImportExportSample
            case TaskType.INSTANCE_SEGMENTATION:
                return InstanceSegmentationImportExportSample
            case _:
                raise ValueError(f"Unknown task type: {task}")
