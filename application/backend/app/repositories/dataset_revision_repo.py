# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from datetime import datetime
from typing import cast

from sqlalchemy import CursorResult, delete, exists, func, select
from sqlalchemy.orm import Session

from app.db.schema import DatasetItemDB, DatasetRevisionDB, DatasetViewDB, DatasetViewItemDB, MediaDB
from app.repositories.base import BaseRepository


class DatasetRevisionRepository(BaseRepository[DatasetRevisionDB]):
    """Repository for dataset revision-related database operations."""

    def __init__(self, project_id: str, db: Session):
        super().__init__(db, DatasetRevisionDB)
        self.project_id = project_id

    def list_all(self) -> Sequence[DatasetRevisionDB]:
        """
        List all dataset revisions for a given project.

        Returns:
            Sequence[DatasetRevisionDB]: A list of dataset revisions associated with the project.
        """
        stmt = select(DatasetRevisionDB).where(DatasetRevisionDB.project_id == self.project_id)
        return self.db.execute(stmt).scalars().all()

    def get_by_id(self, obj_id: str) -> DatasetRevisionDB | None:
        """Get a dataset revision by its id."""
        stmt = select(DatasetRevisionDB).where(
            (DatasetRevisionDB.id == obj_id) & (DatasetRevisionDB.project_id == self.project_id)
        )
        return self.db.execute(stmt).scalar_one_or_none()

    def get_latest_uptodate_dataset_revision(self, dataset_view_id: str | None = None) -> DatasetRevisionDB | None:
        """
        Get latest up to date created dataset revision with the given scope, if it exists.

        Up to date means the revision was created after the last change to the data it covers. For a
        full-dataset revision (``dataset_view_id=None``) that is the last update on any dataset item of the
        project. For a view-scoped revision it is the last update on any dataset item belonging to the view,
        or the last change of the view membership itself (tracked by ``DatasetViewDB.updated_at``).

        Args:
            dataset_view_id: Restrict the lookup to revisions created from this dataset view. When None,
                only full-dataset revisions are considered.
        """
        last_data_change = self._get_last_data_change(dataset_view_id)
        if last_data_change is None:
            return None

        scope_filter = (
            DatasetRevisionDB.dataset_view_id.is_(None)
            if dataset_view_id is None
            else DatasetRevisionDB.dataset_view_id == dataset_view_id
        )
        stmt = (
            select(DatasetRevisionDB)
            .where(
                DatasetRevisionDB.project_id == self.project_id,
                scope_filter,
                DatasetRevisionDB.created_at > last_data_change,
            )
            .order_by(DatasetRevisionDB.created_at.desc())
            .limit(1)
        )
        return self.db.execute(stmt).scalar_one_or_none()

    def _get_last_data_change(self, dataset_view_id: str | None) -> datetime | None:
        """Timestamp of the most recent change affecting the data covered by a revision of the given scope."""
        items_stmt = select(func.max(DatasetItemDB.updated_at)).where(DatasetItemDB.project_id == self.project_id)
        if dataset_view_id is None:
            return self.db.scalar(items_stmt)

        view_updated_at = self.db.scalar(
            select(DatasetViewDB.updated_at).where(
                DatasetViewDB.id == dataset_view_id, DatasetViewDB.project_id == self.project_id
            )
        )
        if view_updated_at is None:  # the view does not exist (anymore)
            return None

        items_stmt = items_stmt.join(MediaDB, MediaDB.id == DatasetItemDB.id).where(
            exists(
                select(DatasetViewItemDB.media_id).where(
                    DatasetViewItemDB.dataset_view_id == dataset_view_id,
                    (DatasetViewItemDB.media_id == MediaDB.id) | (DatasetViewItemDB.media_id == MediaDB.video_id),
                )
            ).correlate(MediaDB)
        )
        last_item_change = self.db.scalar(items_stmt)
        return max(last_item_change, view_updated_at) if last_item_change is not None else view_updated_at

    def delete(self, obj_id: str) -> bool:
        """Delete a dataset revision by its id."""
        stmt = delete(DatasetRevisionDB).where(
            (DatasetRevisionDB.id == obj_id) & (DatasetRevisionDB.project_id == self.project_id)
        )
        result = cast(CursorResult, self.db.execute(stmt))
        return result.rowcount > 0
