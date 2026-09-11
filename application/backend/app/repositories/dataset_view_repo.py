# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections import Counter
from collections.abc import Sequence
from datetime import datetime
from typing import Any, cast

from sqlalchemy import ColumnElement, CursorResult, Select, delete, exists, func, select
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import Session

from app.db.schema import DatasetItemDB, DatasetItemLabelDB, DatasetViewDB, DatasetViewItemDB, MediaDB
from app.models.dataset_item import DatasetItemSortBy
from app.models.media import MediaSortBy, MediaType, SortDirection

from .base import BaseRepository
from .filters import (
    _apply_annotation_status_filter,
    _apply_annotation_status_filter_with_video_support,
    _apply_date_range_filter,
    _apply_label_filter_with_video_support,
    _apply_subset_filter,
    _apply_subset_filter_with_video_support,
)

# Maps each sortable field to its underlying DB column. Extend these when a new sort option is introduced.
_MEDIA_SORT_BY_COLUMN = {MediaSortBy.UPLOAD_DATE: MediaDB.created_at}
_ITEM_SORT_BY_COLUMN = {DatasetItemSortBy.CREATION_DATE: DatasetItemDB.created_at}


class DatasetViewRepository(BaseRepository[DatasetViewDB]):
    """Repository for dataset view-related database operations."""

    def __init__(self, project_id: str, db: Session) -> None:
        super().__init__(db, DatasetViewDB)
        self.project_id = project_id

    def get_by_id(self, obj_id: str) -> DatasetViewDB | None:
        """Get a dataset view by its ID, scoped to the repository's project."""
        stmt = select(DatasetViewDB).where(DatasetViewDB.id == obj_id, DatasetViewDB.project_id == self.project_id)
        return self.db.scalar(stmt)

    def list_by_project(self) -> Sequence[DatasetViewDB]:
        """List all dataset views belonging to the repository's project."""
        stmt = select(DatasetViewDB).where(DatasetViewDB.project_id == self.project_id)
        return self.db.scalars(stmt).all()

    def delete(self, obj_id: str) -> bool:
        """Delete a dataset view by its ID, scoped to the repository's project."""
        stmt = delete(DatasetViewDB).where(DatasetViewDB.id == obj_id, DatasetViewDB.project_id == self.project_id)
        result = cast(CursorResult, self.db.execute(stmt))
        return result.rowcount > 0

    def assign_media(self, dataset_view_id: str, media_ids: list[str]) -> None:
        """Assign one or more media items to a dataset view. Already-assigned items are silently ignored."""
        if not media_ids:
            return
        values = [{"dataset_view_id": dataset_view_id, "media_id": media_id} for media_id in media_ids]
        stmt = insert(DatasetViewItemDB).values(values).on_conflict_do_nothing()
        self.db.execute(stmt)

    def unassign_media(self, dataset_view_id: str, media_ids: list[str]) -> None:
        """Unassign one or more media items from a dataset view. Non-assigned items are silently ignored."""
        if not media_ids:
            return
        stmt = delete(DatasetViewItemDB).where(
            DatasetViewItemDB.dataset_view_id == dataset_view_id,
            DatasetViewItemDB.media_id.in_(media_ids),
        )
        self.db.execute(stmt)

    def _media_in_view_condition(self, dataset_view_id: str) -> ColumnElement[bool]:
        """
        SQLAlchemy EXISTS condition matching ``MediaDB`` rows that belong to the given dataset view.

        A media item is considered to belong to a view either directly (images and videos are assigned to a
        view as a whole) or via its parent video (video frames are never assigned individually, but are
        considered part of the view if the video they belong to is assigned).

        NOTE: This condition must be used against a query that has ``MediaDB`` in its FROM clause, since it
        correlates against it.
        """
        return exists(
            select(DatasetViewItemDB.media_id).where(
                DatasetViewItemDB.dataset_view_id == dataset_view_id,
                (DatasetViewItemDB.media_id == MediaDB.id) | (DatasetViewItemDB.media_id == MediaDB.video_id),
            )
        ).correlate(MediaDB)

    def _base_media_select(self, dataset_view_id: str) -> Select:
        """Media items directly assigned to the view (images and videos; frames are never assigned individually)."""
        return (
            select(MediaDB)
            .join(DatasetViewItemDB, DatasetViewItemDB.media_id == MediaDB.id)
            .where(
                DatasetViewItemDB.dataset_view_id == dataset_view_id,
                MediaDB.project_id == self.project_id,
                MediaDB.type != MediaType.VIDEO_FRAME,
            )
        )

    def count_media(
        self,
        dataset_view_id: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        annotation_status: str | None = None,
        label_ids: list[str] | None = None,
        subsets: list[str] | None = None,
    ) -> int:
        """Count the media items (images and videos) assigned to a dataset view, optionally matching filters."""
        stmt = (
            select(func.count(func.distinct(MediaDB.id)))
            .select_from(MediaDB)
            .join(DatasetViewItemDB, DatasetViewItemDB.media_id == MediaDB.id)
            .join(DatasetItemDB, DatasetItemDB.id == MediaDB.id, isouter=True)
            .where(
                DatasetViewItemDB.dataset_view_id == dataset_view_id,
                MediaDB.project_id == self.project_id,
                MediaDB.type != MediaType.VIDEO_FRAME,
            )
        )
        stmt = _apply_date_range_filter(stmt, MediaDB.created_at, start_date, end_date)
        stmt = _apply_annotation_status_filter_with_video_support(stmt, annotation_status)
        stmt = _apply_subset_filter_with_video_support(stmt, subsets)
        stmt = _apply_label_filter_with_video_support(stmt, label_ids)
        return self.db.scalar(stmt) or 0

    def list_media(  # noqa: PLR0913
        self,
        dataset_view_id: str,
        limit: int,
        offset: int,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        annotation_status: str | None = None,
        label_ids: list[str] | None = None,
        subsets: list[str] | None = None,
        sort_by: MediaSortBy = MediaSortBy.UPLOAD_DATE,
        sort_direction: SortDirection = SortDirection.DESC,
    ) -> list[MediaDB]:
        """List (filter) the media items assigned to a dataset view."""
        stmt = self._base_media_select(dataset_view_id).join(
            DatasetItemDB, DatasetItemDB.id == MediaDB.id, isouter=True
        )
        stmt = _apply_date_range_filter(stmt, MediaDB.created_at, start_date, end_date)
        stmt = _apply_annotation_status_filter_with_video_support(stmt, annotation_status)
        stmt = _apply_subset_filter_with_video_support(stmt, subsets)
        stmt = _apply_label_filter_with_video_support(stmt, label_ids)
        sort_column = _MEDIA_SORT_BY_COLUMN[sort_by]
        order_by_column = sort_column.asc() if sort_direction == SortDirection.ASC else sort_column.desc()
        stmt = stmt.order_by(order_by_column).offset(offset).limit(limit)
        return list(self.db.scalars(stmt).all())

    def _base_item_select(self, dataset_view_id: str) -> Select:
        """
        Dataset items belonging to media assigned to the view.

        This includes items for images assigned directly to the view, as well as items for video frames
        whose parent video is assigned to the view.
        """
        return (
            select(DatasetItemDB)
            .join(MediaDB, MediaDB.id == DatasetItemDB.id)
            .where(
                DatasetItemDB.project_id == self.project_id,
                self._media_in_view_condition(dataset_view_id),
            )
        )

    def count_items(
        self,
        dataset_view_id: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        annotation_status: str | None = None,
        label_ids: list[str] | None = None,
        subsets: list[str] | None = None,
    ) -> int:
        """Count the dataset items assigned to a dataset view, optionally matching the given filters."""
        select_fn = func.count(func.distinct(DatasetItemDB.id)) if label_ids else func.count()
        stmt = (
            select(select_fn)
            .select_from(DatasetItemDB)
            .join(MediaDB, MediaDB.id == DatasetItemDB.id)
            .where(
                DatasetItemDB.project_id == self.project_id,
                self._media_in_view_condition(dataset_view_id),
            )
        )
        stmt = _apply_date_range_filter(stmt, DatasetItemDB.created_at, start_date, end_date)
        stmt = _apply_annotation_status_filter(stmt, annotation_status)
        stmt = _apply_subset_filter(stmt, subsets)
        if label_ids:
            stmt = stmt.join(DatasetItemLabelDB).where(DatasetItemLabelDB.label_id.in_(label_ids))
        return self.db.scalar(stmt) or 0

    def list_items(  # noqa: PLR0913
        self,
        dataset_view_id: str,
        limit: int,
        offset: int,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        annotation_status: str | None = None,
        label_ids: list[str] | None = None,
        subsets: list[str] | None = None,
        sort_by: DatasetItemSortBy = DatasetItemSortBy.CREATION_DATE,
        sort_direction: SortDirection = SortDirection.DESC,
    ) -> list[DatasetItemDB]:
        """List (filter) the dataset items assigned to a dataset view."""
        stmt = self._base_item_select(dataset_view_id)
        stmt = _apply_date_range_filter(stmt, DatasetItemDB.created_at, start_date, end_date)
        stmt = _apply_annotation_status_filter(stmt, annotation_status)
        stmt = _apply_subset_filter(stmt, subsets)
        if label_ids:
            stmt = stmt.join(DatasetItemLabelDB).where(DatasetItemLabelDB.label_id.in_(label_ids)).distinct()
        sort_column = _ITEM_SORT_BY_COLUMN[sort_by]
        order_by_column = sort_column.asc() if sort_direction == SortDirection.ASC else sort_column.desc()
        stmt = stmt.order_by(order_by_column).offset(offset).limit(limit)
        return list(self.db.scalars(stmt).all())

    def list_items_with_media(  # noqa: PLR0913
        self,
        dataset_view_id: str,
        limit: int,
        offset: int,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        annotation_status: str | None = None,
        label_ids: list[str] | None = None,
        subsets: list[str] | None = None,
    ) -> list[tuple[DatasetItemDB, MediaDB]]:
        """List (filter) the dataset items assigned to a dataset view, together with their media."""
        stmt = (
            select(DatasetItemDB, MediaDB)
            .join(MediaDB, MediaDB.id == DatasetItemDB.id)
            .where(
                DatasetItemDB.project_id == self.project_id,
                self._media_in_view_condition(dataset_view_id),
            )
        )
        stmt = _apply_date_range_filter(stmt, DatasetItemDB.created_at, start_date, end_date)
        stmt = _apply_annotation_status_filter(stmt, annotation_status)
        stmt = _apply_subset_filter(stmt, subsets)
        if label_ids:
            stmt = stmt.join(DatasetItemLabelDB).where(DatasetItemLabelDB.label_id.in_(label_ids)).distinct()
        stmt = stmt.order_by(DatasetItemDB.created_at.desc()).offset(offset).limit(limit)
        return [(dataset_item, media) for (dataset_item, media) in self.db.execute(stmt).all()]

    def get_statistics(self, dataset_view_id: str) -> dict[str, Any]:
        """Get statistics (media & annotation counts) about the media assigned to a dataset view."""
        # Media Counts (images, videos, video frames) among media directly assigned to the view:
        media_counts_stmt = (
            select(
                MediaDB.type,
                func.count(MediaDB.id).label("count"),
                func.sum(MediaDB.frame_count).label("frame_count"),
            )
            .join(DatasetViewItemDB, DatasetViewItemDB.media_id == MediaDB.id)
            .where(
                DatasetViewItemDB.dataset_view_id == dataset_view_id,
                MediaDB.project_id == self.project_id,
            )
            .group_by(MediaDB.type)
        )
        rows = list(self.db.execute(media_counts_stmt))
        statistics: dict[str, Any] = {f"{row.type}s": cast(int, row.count) for row in rows}
        statistics["video_frames"] = int(sum(row.frame_count or 0 for row in rows))

        # Annotation Counts (annotated images and video frames belonging to media assigned to the view):
        annotated_images_frames_stmt = (
            select(MediaDB.type, func.count(DatasetItemDB.id).label("count"))
            .join(DatasetItemDB, DatasetItemDB.id == MediaDB.id)
            .where(
                DatasetItemDB.project_id == self.project_id,
                DatasetItemDB.annotation_data.isnot(None),
                DatasetItemDB.user_reviewed,
                self._media_in_view_condition(dataset_view_id),
            )
            .group_by(MediaDB.type)
        )
        annotated_counts: dict[str, Any] = {
            f"annotated_{row.type}s": row.count for row in self.db.execute(annotated_images_frames_stmt)
        }

        # Annotation Counts (annotated videos assigned to the view)
        annotated_video_stmt = (
            select(func.count(func.distinct(MediaDB.video_id)))
            .join(DatasetItemDB, DatasetItemDB.id == MediaDB.id)
            .where(
                DatasetItemDB.project_id == self.project_id,
                DatasetItemDB.annotation_data.isnot(None),
                DatasetItemDB.user_reviewed,
                MediaDB.type == MediaType.VIDEO_FRAME,
                self._media_in_view_condition(dataset_view_id),
            )
        )
        annotated_counts["annotated_videos"] = self.db.scalar(annotated_video_stmt) or 0

        # Total instances and instances_per_label
        annotated_dataset_items_stmt = (
            select(DatasetItemDB.annotation_data)
            .join(MediaDB, MediaDB.id == DatasetItemDB.id)
            .where(
                DatasetItemDB.project_id == self.project_id,
                DatasetItemDB.annotation_data.isnot(None),
                DatasetItemDB.user_reviewed,
                self._media_in_view_condition(dataset_view_id),
            )
        )
        total_instances = 0
        labels_counts: Counter[str] = Counter()
        no_object_count = 0
        for item in self.db.execute(annotated_dataset_items_stmt):
            if not item.annotation_data:
                no_object_count += 1
            else:
                total_instances += len(item.annotation_data)
                for annotation in item.annotation_data:
                    for label in annotation["labels"]:
                        labels_counts[label["id"]] += 1

        annotated_counts["instances"] = total_instances
        annotated_counts["instances_per_label"] = [
            {"label_id": label_id, "instances": count} for label_id, count in labels_counts.items()
        ]
        annotated_counts["instances_per_label"].append({"label_id": None, "instances": no_object_count})  # type: ignore[dict-item]

        statistics.update(annotated_counts)

        return statistics
