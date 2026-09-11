# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from datumaro.experimental import Dataset
from sqlalchemy.orm import Session

from app.datumaro_converter import SampleMode
from app.db.schema import DatasetItemDB, MediaDB
from app.models import (
    DatasetItem,
    DatasetItemAnnotation,
    DatasetItemAnnotationStatus,
    DatasetItemSubset,
    FullImage,
    Label,
    Media,
    Polygon,
    Project,
    Rectangle,
    Task,
    TaskType,
)
from app.models.dataset import DatasetStatistics
from app.models.dataset_item import DatasetItemSortBy
from app.models.media import MediaAdapter, SortDirection, VideoFrame
from app.repositories import DatasetItemRepository, DatasetViewRepository
from app.services.media_service import MediaService

from .base import BaseSessionManagedService, ResourceNotFoundError, ResourceType
from .label_service import LabelService

DEFAULT_THUMBNAIL_SIZE = 256


class AnnotationValidationError(Exception):
    """Exception raised when dataset annotation validation has failed."""

    def __init__(self, message: str):
        super().__init__(message)


class SubsetAlreadyAssignedError(Exception):
    """Exception raised when subset is being assigned to a dataset item, which already has one assigned."""

    def __init__(self, message: str | None = None):
        msg = message or "Dataset item has already a subset assigned."
        super().__init__(msg)


@dataclass(frozen=True)
class DatasetItemFilters:
    limit: int = 20
    offset: int = 0
    start_date: datetime | None = None
    end_date: datetime | None = None
    annotation_status: DatasetItemAnnotationStatus | None = None
    label_ids: list[UUID] | None = None
    subsets: list[str] | None = None
    sort_by: DatasetItemSortBy = DatasetItemSortBy.CREATION_DATE
    sort_direction: SortDirection = SortDirection.DESC


class DatasetService(BaseSessionManagedService):
    def __init__(
        self,
        label_service: LabelService,
        media_service: MediaService,
        db_session: Session | None = None,
    ) -> None:
        super().__init__(db_session)

        self._label_service = label_service
        self._media_service = media_service
        self.register_managed_services(label_service, media_service)

    def create_dataset_item(
        self,
        project_id: UUID,
        task: Task,
        media: Media,
        user_reviewed: bool,
        prediction_model_id: UUID | None = None,
        annotations: list[DatasetItemAnnotation] | None = None,
        subset: DatasetItemSubset = DatasetItemSubset.UNASSIGNED,
    ) -> DatasetItem:
        """Creates a new dataset item"""

        dataset_item = DatasetItemDB(
            id=str(media.id),
            project_id=str(project_id),
            subset=subset,
            user_reviewed=user_reviewed,
            prediction_model_id=str(prediction_model_id) if prediction_model_id is not None else None,
        )

        if annotations is not None:
            labels = self._label_service.list_all(project_id=project_id)
            annotations = DatasetService._cleanup_and_validate_annotations(
                annotations=annotations, task=task, labels=labels, media=media, user_reviewed=user_reviewed
            )

            dataset_item.annotation_data = [annotation.model_dump(mode="json") for annotation in annotations]

        repo = DatasetItemRepository(project_id=str(project_id), db=self.db_session)
        db_dataset_item = repo.save(dataset_item)
        if annotations is not None:
            repo.set_labels(
                dataset_item_id=str(media.id),
                label_ids={str(label.id) for annotation in annotations for label in annotation.labels},
            )
        return DatasetItem.model_validate(db_dataset_item)

    def count_dataset_items(
        self,
        project: Project,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        annotation_status: DatasetItemAnnotationStatus | None = None,
        label_ids: list[UUID] | None = None,
        subsets: list[str] | None = None,
    ) -> int:
        """Get number of available dataset items (within date range if specified)"""
        repo = DatasetItemRepository(project_id=str(project.id), db=self.db_session)
        label_ids_str = [str(label_id) for label_id in label_ids] if label_ids else None
        return repo.count(
            start_date=start_date,
            end_date=end_date,
            annotation_status=annotation_status,
            label_ids=label_ids_str,
            subsets=subsets,
        )

    def get_dataset_statistics(self, project_id: UUID) -> DatasetStatistics:
        """Get dataset statistics"""
        repo = DatasetItemRepository(project_id=str(project_id), db=self.db_session)
        statistics_dict = repo.get_statistics()
        return DatasetStatistics.model_validate(statistics_dict)

    def list_dataset_items(
        self,
        project_id: UUID,
        filters: DatasetItemFilters | None = None,
    ) -> list[DatasetItem]:
        """Get information about available dataset items"""
        if filters is None:
            filters = DatasetItemFilters()
        repo = DatasetItemRepository(project_id=str(project_id), db=self.db_session)
        label_ids_str = [str(label_id) for label_id in filters.label_ids] if filters.label_ids else None
        return [
            DatasetItem.model_validate(db)
            for db in repo.list_items(
                limit=filters.limit,
                offset=filters.offset,
                start_date=filters.start_date,
                end_date=filters.end_date,
                annotation_status=filters.annotation_status,
                label_ids=label_ids_str,
                subsets=filters.subsets,
                sort_by=filters.sort_by,
                sort_direction=filters.sort_direction,
            )
        ]

    def list_dataset_items_with_media(
        self,
        project_id: UUID,
        filters: DatasetItemFilters | None = None,
    ) -> list[tuple[DatasetItem, Media]]:
        """Get information about available dataset items with corresponding media info"""
        if filters is None:
            filters = DatasetItemFilters()
        repo = DatasetItemRepository(project_id=str(project_id), db=self.db_session)
        label_ids_str = [str(label_id) for label_id in filters.label_ids] if filters.label_ids else None
        return [
            (DatasetItem.model_validate(db_dataset_item), MediaAdapter.validate_python(db_media))
            for db_dataset_item, db_media in repo.list_items_with_media(
                limit=filters.limit,
                offset=filters.offset,
                start_date=filters.start_date,
                end_date=filters.end_date,
                annotation_status=filters.annotation_status,
                label_ids=label_ids_str,
                subsets=filters.subsets,
            )
        ]

    def get_dataset_item_by_id(self, project_id: UUID, dataset_item_id: UUID) -> DatasetItem:
        """Get a dataset item by its ID"""
        repo = DatasetItemRepository(project_id=str(project_id), db=self.db_session)
        db_dataset_item = repo.get_by_id(str(dataset_item_id))
        if not db_dataset_item:
            raise ResourceNotFoundError(ResourceType.DATASET_ITEM, str(dataset_item_id))
        return DatasetItem.model_validate(db_dataset_item)

    @staticmethod
    def _cleanup_and_validate_annotations(
        annotations: list[DatasetItemAnnotation],
        task: Task,
        labels: list[Label],
        media: Media,
        user_reviewed: bool,
    ) -> list[DatasetItemAnnotation]:
        if user_reviewed:
            # if user reviewed, user has accepted all predictions and/or added new annotations,
            # so confidence scores are no longer meaningful
            annotations = [annotation.model_copy(update={"confidences": None}) for annotation in annotations]

        DatasetService._validate_annotations_labels(annotations=annotations, labels=labels)
        DatasetService._validate_annotation_shapes(annotations=annotations, task=task)
        DatasetService._validate_annotations_coordinates(annotations=annotations, media=media)

        return annotations

    @staticmethod
    def _validate_annotations_labels(annotations: list[DatasetItemAnnotation], labels: Sequence[Label]) -> None:
        for annotation in annotations:
            for annotation_label in annotation.labels:
                project_label = next((label for label in labels if label.id == annotation_label.id), None)
                if project_label is None:
                    raise AnnotationValidationError(f"Label {str(annotation_label.id)} is not found in the project.")

    @staticmethod
    def _validate_annotation_shapes(annotations: list[DatasetItemAnnotation], task: Task) -> None:  # noqa: C901, PLR0912
        match task.task_type:
            case TaskType.CLASSIFICATION:
                if len(annotations) == 0:
                    if task.exclusive_labels:  # multiclass classification -> empty label not allowed
                        raise AnnotationValidationError("Multiclass classification project requires one annotation.")
                    # multilabel classification -> empty label allowed
                    return
                if len(annotations) > 1:
                    raise AnnotationValidationError("Classification project doesn't allow more than one annotation.")
                annotation = annotations[0]
                if not isinstance(annotation.shape, FullImage):
                    raise AnnotationValidationError("Classification project supports only full_image shapes.")
                if task.exclusive_labels and len(annotation.labels) > 1:
                    raise AnnotationValidationError(
                        "Multiclass classification project doesn't allow more than one label per annotation."
                    )
            case TaskType.DETECTION:
                for annotation in annotations:
                    if not isinstance(annotation.shape, Rectangle):
                        raise AnnotationValidationError("Detection project supports only rectangle shapes.")
                    if len(annotation.labels) > 1:
                        raise AnnotationValidationError(
                            "Detection project doesn't allow more than one label per annotation."
                        )
            case TaskType.INSTANCE_SEGMENTATION:
                for annotation in annotations:
                    if not isinstance(annotation.shape, Polygon):
                        raise AnnotationValidationError("Instance Segmentation project supports only polygon shapes.")
                    if len(annotation.labels) > 1:
                        raise AnnotationValidationError(
                            "Segmentation project doesn't allow more than one label per annotation."
                        )

    @staticmethod
    def _validate_annotations_coordinates(annotations: list[DatasetItemAnnotation], media: Media | MediaDB) -> None:
        for annotation in annotations:
            if isinstance(annotation.shape, Rectangle):
                rect = annotation.shape
                x1, x2 = rect.x, rect.x + rect.width
                if x1 > media.width or x2 > media.width:
                    raise AnnotationValidationError(
                        f"Rectangle coordinates (x1={x1}, x2={x2}) are out of bounds for media width {media.width}"
                    )
                y1, y2 = rect.y, rect.y + rect.height
                if y1 > media.height or y2 > media.height:
                    raise AnnotationValidationError(
                        f"Rectangle coordinates (y1={y1}, y2={y2}) are out of bounds for media height {media.height}"
                    )
            if isinstance(annotation.shape, Polygon):
                poly = annotation.shape
                for point in poly.points:
                    if point.x > media.width or point.y > media.height:
                        raise AnnotationValidationError(
                            f"Polygon points (x={point.x}, y={point.y}) are out of bounds for media "
                            f"({media.width}, {media.height})"
                        )

    def set_dataset_item_annotations(
        self,
        project: Project,
        dataset_item_id: UUID,
        annotations: list[DatasetItemAnnotation],
        user_reviewed: bool,
        prediction_model_id: UUID | None,
    ) -> DatasetItem:
        """
        Set dataset item annotations

        Args:
            project: The project to which the dataset item belongs.
            dataset_item_id: The ID of the dataset item.
            annotations: The list of annotations to set. Overwrites existing annotations, if any.
            user_reviewed: Whether the annotations have been reviewed by a user.
            prediction_model_id: Identifier of the model that generated predictions for this
                dataset item, if applicable.

        Returns:
            The updated dataset item.
        """
        labels = self._label_service.list_all(project_id=project.id)
        media = self._media_service.get_media_by_id(project_id=project.id, media_id=dataset_item_id)
        annotations = DatasetService._cleanup_and_validate_annotations(
            annotations=annotations, task=project.task, labels=labels, media=media, user_reviewed=user_reviewed
        )

        repo = DatasetItemRepository(project_id=str(project.id), db=self.db_session)
        if not repo.set_annotation_data(
            obj_id=str(dataset_item_id),
            annotation_data=[annotation.model_dump(mode="json") for annotation in annotations],
            user_reviewed=user_reviewed,
            prediction_model_id=str(prediction_model_id) if prediction_model_id is not None else None,
        ):
            raise ResourceNotFoundError(ResourceType.DATASET_ITEM, str(dataset_item_id))

        repo.set_labels(
            dataset_item_id=str(dataset_item_id),
            label_ids={str(label.id) for annotation in annotations for label in annotation.labels},
        )
        return self.get_dataset_item_by_id(project_id=project.id, dataset_item_id=dataset_item_id)

    def delete_dataset_item_annotations(self, project: Project, dataset_item_id: UUID) -> None:
        """Delete the dataset item annotations"""
        repo = DatasetItemRepository(project_id=str(project.id), db=self.db_session)
        updated = repo.delete_annotation_data(obj_id=str(dataset_item_id))
        if not updated:
            raise ResourceNotFoundError(ResourceType.DATASET_ITEM, str(dataset_item_id))
        repo.delete_labels(dataset_item_id=str(dataset_item_id))

    def assign_dataset_item_subset(
        self, project_id: UUID, dataset_item_id: UUID, subset: DatasetItemSubset
    ) -> DatasetItem:
        """Assign dataset item subset"""
        repo = DatasetItemRepository(project_id=str(project_id), db=self.db_session)
        db_subset = repo.get_subset(str(dataset_item_id))
        if db_subset is None:
            raise ResourceNotFoundError(ResourceType.DATASET_ITEM, str(dataset_item_id))
        if db_subset == DatasetItemSubset.UNASSIGNED:
            repo.set_subset(obj_ids={str(dataset_item_id)}, subset=subset)
        elif db_subset != subset:
            raise SubsetAlreadyAssignedError
        # If db_subset == subset, it's a no-op (same subset already assigned)

        return self.get_dataset_item_by_id(project_id=project_id, dataset_item_id=dataset_item_id)

    def get_dm_dataset(
        self,
        project_id: UUID,
        task: Task,
        annotation_status: DatasetItemAnnotationStatus | None,
        sample_mode: SampleMode,
        dataset_view_id: UUID | None = None,
    ) -> Dataset:
        from app.datumaro_converter import SampleMode, convert_dataset

        view_repo = DatasetViewRepository(project_id=str(project_id), db=self.db_session)
        if dataset_view_id is not None and view_repo.get_by_id(str(dataset_view_id)) is None:
            raise ResourceNotFoundError(ResourceType.DATASET_VIEW, str(dataset_view_id))

        def get_dataset_items_and_media(offset: int, limit: int) -> list[tuple[DatasetItem, Media]]:
            if dataset_view_id is not None:
                return [
                    (DatasetItem.model_validate(db_dataset_item), MediaAdapter.validate_python(db_media))
                    for db_dataset_item, db_media in view_repo.list_items_with_media(
                        dataset_view_id=str(dataset_view_id),
                        limit=limit,
                        offset=offset,
                        annotation_status=annotation_status,
                    )
                ]
            return self.list_dataset_items_with_media(
                project_id=project_id,
                filters=DatasetItemFilters(limit=limit, offset=offset, annotation_status=annotation_status),
            )

        def _get_media_path(media: Media) -> str:
            """
            Returns the media path used to construct the sample during conversion.

            Video frames for import/export samples should use the video binary path to ensure that the frame
            information is preserved in the sample, while for other media types or training samples, the media binary
            path is sufficient.
            """
            if isinstance(media, VideoFrame) and sample_mode == SampleMode.IMPORT_EXPORT:
                return str(
                    self._media_service.get_media_binary_path_by_id(project_id=project_id, media_id=media.video_id)
                )
            return str(self._media_service.get_media_binary_path_by_id(project_id=project_id, media_id=media.id))

        labels = self._label_service.list_all(project_id=project_id)
        return convert_dataset(
            task=task,
            labels=labels,
            get_dataset_items_and_media=get_dataset_items_and_media,
            get_media_path=_get_media_path,
            sample_mode=sample_mode,
        )
