# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from datetime import UTC, datetime
from typing import cast

from sqlalchemy import CursorResult, Select, delete, func, select
from sqlalchemy.orm import Session

from app.db.schema import DatasetItemDB, MediaDB
from app.models import MediaType
from app.models.media import MediaSortBy, SortDirection

from .filters import (
    _apply_annotation_status_filter_with_video_support,
    _apply_label_filter_with_video_support,
    _apply_subset_filter_with_video_support,
)

# Maps each sortable field to its underlying DB column. Extend this when a new
# `MediaSortBy` member is introduced.
_SORT_BY_COLUMN = {
    MediaSortBy.UPLOAD_DATE: MediaDB.created_at,
}


class MediaRepository:
    """Repository for media-related database operations."""

    def __init__(self, project_id: str, db: Session):
        self.project_id = project_id
        self.db = db

    def _base_select(self) -> Select:
        """Create base select statement filtered by project_id."""
        return select(MediaDB).where(MediaDB.project_id == self.project_id)

    def _base_id_select(self) -> Select:
        """Create base select statement of only id/type, filtered by project_id."""
        return select(MediaDB.id, MediaDB.type).where(MediaDB.project_id == self.project_id)

    @staticmethod
    def _apply_date_filters(
        stmt: Select, start_date: datetime | None = None, end_date: datetime | None = None
    ) -> Select:
        """Apply date range filters to a select statement."""
        if start_date:
            stmt = stmt.where(MediaDB.created_at >= start_date)
        if end_date:
            stmt = stmt.where(MediaDB.created_at < end_date)
        return stmt

    def save(self, media_db: MediaDB) -> MediaDB:
        media_db.updated_at = datetime.now(UTC)
        self.db.add(media_db)
        self.db.flush()
        return media_db

    def count(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        annotation_status: str | None = None,
        label_ids: list[str] | None = None,
        subsets: list[str] | None = None,
        exclude_types: list[MediaType] | None = None,
    ) -> int:
        stmt = (
            select(func.count(func.distinct(MediaDB.id)))
            .select_from(MediaDB)
            .join(
                DatasetItemDB,
                DatasetItemDB.id == MediaDB.id,
                isouter=True,
            )
            .where(MediaDB.project_id == self.project_id)
        )
        if exclude_types:
            stmt = stmt.where(MediaDB.type.not_in(exclude_types))
        stmt = self._apply_date_filters(stmt, start_date, end_date)
        stmt = _apply_annotation_status_filter_with_video_support(stmt, annotation_status)
        stmt = _apply_subset_filter_with_video_support(stmt, subsets)
        stmt = _apply_label_filter_with_video_support(stmt, label_ids)
        return self.db.scalar(stmt) or 0

    def list_items(  # noqa: PLR0913
        self,
        limit: int,
        offset: int,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        annotation_status: str | None = None,
        label_ids: list[str] | None = None,
        subsets: list[str] | None = None,
        exclude_types: list[MediaType] | None = None,
        sort_by: MediaSortBy = MediaSortBy.UPLOAD_DATE,
        sort_direction: SortDirection = SortDirection.DESC,
    ) -> list[MediaDB]:
        stmt = self._base_select().join(DatasetItemDB, DatasetItemDB.id == MediaDB.id, isouter=True)
        if exclude_types:
            stmt = stmt.where(MediaDB.type.not_in(exclude_types))
        stmt = self._apply_date_filters(stmt, start_date, end_date)
        stmt = _apply_annotation_status_filter_with_video_support(stmt, annotation_status)
        stmt = _apply_subset_filter_with_video_support(stmt, subsets)
        stmt = _apply_label_filter_with_video_support(stmt, label_ids)
        sort_column = _SORT_BY_COLUMN[sort_by]
        order_by_column = sort_column.asc() if sort_direction == SortDirection.ASC else sort_column.desc()
        stmt = stmt.order_by(order_by_column).offset(offset).limit(limit)
        return list(self.db.scalars(stmt).all())

    def list_item_ids(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        annotation_status: str | None = None,
        label_ids: list[str] | None = None,
        subsets: list[str] | None = None,
        exclude_types: list[MediaType] | None = None,
    ) -> tuple[tuple[str, str], ...]:
        """Get the (id, type) of every media item matching the given filters, without loading full rows."""
        stmt = self._base_id_select().join(DatasetItemDB, DatasetItemDB.id == MediaDB.id, isouter=True)
        if exclude_types:
            stmt = stmt.where(MediaDB.type.not_in(exclude_types))
        stmt = self._apply_date_filters(stmt, start_date, end_date)
        stmt = _apply_annotation_status_filter_with_video_support(stmt, annotation_status)
        stmt = _apply_subset_filter_with_video_support(stmt, subsets)
        stmt = _apply_label_filter_with_video_support(stmt, label_ids)
        return tuple((media_id, media_type) for media_id, media_type in self.db.execute(stmt).all())

    def get_earliest(self) -> MediaDB | None:
        """
        Get the earliest media based on creation date.

        This method efficiently uses the multicolumn index on (project_id, created_at)
        for optimal query performance.

        Returns:
            The earliest MediaDB instance or None if no items exist.
        """
        stmt = self._base_select().order_by(MediaDB.created_at.asc()).limit(1)
        return self.db.scalar(stmt)

    def get_by_id(self, obj_id: str) -> MediaDB | None:
        stmt = self._base_select().where(MediaDB.id == obj_id)
        return self.db.scalar(stmt)

    def get_by_ids(self, obj_ids: list[str]) -> list[MediaDB]:
        stmt = self._base_select().where(MediaDB.id.in_(obj_ids))
        return list(self.db.scalars(stmt).all())

    def delete(self, obj_id: str) -> bool:
        stmt = delete(MediaDB).where(
            MediaDB.project_id == self.project_id,
            MediaDB.id == obj_id,
        )
        result = cast(CursorResult, self.db.execute(stmt))
        return result.rowcount > 0

    def get_video_frame_by_video_id_and_index(self, video_id: str, frame_index: int) -> MediaDB | None:
        stmt = self._base_select().where(MediaDB.video_id == video_id, MediaDB.frame_index == frame_index)
        return self.db.scalar(stmt)

    def search_video_frames_by_video_id_and_indexes(self, video_id: str, frame_indexes: list[int]) -> list[MediaDB]:
        stmt = self._base_select().where(MediaDB.video_id == video_id, MediaDB.frame_index.in_(frame_indexes))
        return list(self.db.scalars(stmt).all())

    def list_annotated_video_frames_by_video_id(
        self,
        video_id: str,
        frame_index_from: int = 0,
        frame_index_to: int = 10,
    ) -> list[tuple[DatasetItemDB, MediaDB]]:
        stmt = (
            select(DatasetItemDB, MediaDB)
            .where(DatasetItemDB.project_id == self.project_id)
            .join(MediaDB)
            .order_by(MediaDB.frame_index.asc())
            .where(
                MediaDB.id == DatasetItemDB.id,
                MediaDB.project_id == DatasetItemDB.project_id,
                MediaDB.video_id == video_id,
                MediaDB.frame_index >= frame_index_from,
                MediaDB.frame_index <= frame_index_to,
            )
        )
        return [(dataset_item, media) for (dataset_item, media) in self.db.execute(stmt).all()]

    def count_annotated_video_frames_by_video_id(self, video_id: str) -> int:
        stmt = (
            select(func.count(MediaDB.id))
            .select_from(MediaDB)
            .join(DatasetItemDB, DatasetItemDB.id == MediaDB.id)
            .where(
                MediaDB.project_id == self.project_id,
                MediaDB.video_id == video_id,
                DatasetItemDB.annotation_data.is_not(None),
            )
        )
        return self.db.scalar(stmt) or 0
