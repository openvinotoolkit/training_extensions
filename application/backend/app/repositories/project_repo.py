# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.schema import ModelRevisionDB, ProjectDB
from app.repositories.base import BaseRepository


class ProjectRepository(BaseRepository[ProjectDB]):
    """Repository for project-related database operations."""

    def __init__(self, db: Session):
        super().__init__(db, ProjectDB)

    def count_model_revisions(self, training_status: str, project_ids: Sequence[str] | None = None) -> dict[str, int]:
        """
        Count the model revisions with the given training status, grouped by project.

        Args:
            training_status (str): Training status to filter on.
            project_ids (Sequence[str] | None): Optional project ids to restrict the count to.

        Returns:
            dict[str, int]: Mapping of project id to model revision count. Projects without matches are omitted.
        """
        stmt = (
            select(ModelRevisionDB.project_id, func.count(ModelRevisionDB.id))
            .where(ModelRevisionDB.training_status == training_status)
            .group_by(ModelRevisionDB.project_id)
        )
        if project_ids is not None:
            stmt = stmt.where(ModelRevisionDB.project_id.in_(project_ids))
        return dict(self.db.execute(stmt).tuples().all())
