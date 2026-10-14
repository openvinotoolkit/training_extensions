# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import shutil
from pathlib import Path
from uuid import UUID

from loguru import logger
from sqlalchemy.orm import Session

from app.db.schema import ProjectDB
from app.models import Label, PipelineStatus, Project, Task, TrainingStatus
from app.repositories import MediaRepository, ProjectRepository
from app.repositories.base import PrimaryKeyIntegrityError

from .base import (
    BaseSessionManagedService,
    ResourceInUseError,
    ResourceNotFoundError,
    ResourceType,
    ResourceWithIdAlreadyExistsError,
)
from .label_service import LabelService
from .parent_process_guard import parent_process_only
from .pipeline_service import PipelineService

MSG_ERR_DELETE_ACTIVE_PROJECT = "Cannot delete a project with a running pipeline."


class ProjectService(BaseSessionManagedService):
    def __init__(
        self,
        data_dir: Path,
        label_service: LabelService,
        pipeline_service: PipelineService,
        db_session: Session | None = None,
    ) -> None:
        super().__init__(db_session)
        self._projects_dir = data_dir / "projects"
        self._label_service: LabelService = label_service
        self._pipeline_service: PipelineService = pipeline_service
        self.register_managed_services(label_service, pipeline_service)

    def create_project(self, project_id: UUID, name: str, task: Task) -> Project:
        project_repo = ProjectRepository(self.db_session)
        try:
            project_db = project_repo.save(
                ProjectDB(
                    id=str(project_id),
                    name=name,
                    task_type=str(task.task_type),
                    exclusive_labels=task.exclusive_labels,
                )
            )
        except PrimaryKeyIntegrityError:
            raise ResourceWithIdAlreadyExistsError(ResourceType.PROJECT, str(project_id))

        pipeline = self._pipeline_service.create_pipeline(project_id=UUID(project_db.id))

        labels: list[Label] = []
        for label in task.labels:
            labels.append(
                self._label_service.create_label(
                    project_id=UUID(project_db.id),
                    name=label.name,
                    color=label.color,
                    hotkey=label.hotkey,
                    label_id=label.id,
                )
            )
        return Project.model_validate(
            {
                **project_db.__dict__,
                "active_pipeline": pipeline.status == PipelineStatus.RUNNING,
                "task": {
                    "task_type": project_db.task_type,
                    "exclusive_labels": project_db.exclusive_labels,
                    "labels": labels,
                },
            }
        )

    def list_projects(self) -> list[Project]:
        project_repo = ProjectRepository(self.db_session)
        return [self._to_project(project_db) for project_db in project_repo.list_all()]

    def get_project_by_id(self, project_id: UUID) -> Project:
        project_repo = ProjectRepository(self.db_session)
        project_db = project_repo.get_by_id(str(project_id))
        if not project_db:
            raise ResourceNotFoundError(ResourceType.PROJECT, str(project_id))
        return self._to_project(project_db)

    def get_trained_model_counts(self, project_ids: list[UUID] | None = None) -> dict[UUID, int]:
        """
        Get the number of successfully trained models per project.

        Args:
            project_ids (list[UUID] | None): Optional project ids to restrict the count to; all projects if None.

        Returns:
            dict[UUID, int]: Mapping of project id to trained model count. Projects without models are omitted.
        """
        project_repo = ProjectRepository(self.db_session)
        counts = project_repo.count_model_revisions(
            training_status=TrainingStatus.SUCCESSFUL,
            project_ids=[str(project_id) for project_id in project_ids] if project_ids is not None else None,
        )
        return {UUID(project_id): count for project_id, count in counts.items()}

    @parent_process_only
    def update_project_name(self, project_id: UUID, name: str) -> Project:
        """Update only the project name"""
        project_repo = ProjectRepository(self.db_session)
        project_db = project_repo.get_by_id(str(project_id))
        if not project_db:
            raise ResourceNotFoundError(ResourceType.PROJECT, str(project_id))
        project_db.name = name
        return self._to_project(project_db)

    @parent_process_only
    def delete_project_by_id(self, project_id: UUID) -> None:
        is_running = self._pipeline_service.is_running(project_id=project_id)
        if is_running:
            raise ResourceInUseError(ResourceType.PROJECT, str(project_id), MSG_ERR_DELETE_ACTIVE_PROJECT)
        project_repo = ProjectRepository(self.db_session)

        project_deleted = project_repo.delete(str(project_id))
        if not project_deleted:
            raise ResourceNotFoundError(ResourceType.PROJECT, str(project_id))

        project_path = self._projects_dir / str(project_id)
        if project_path.exists():
            shutil.rmtree(project_path)
            logger.info("Deleted project files at '{}'", project_path)

    def get_project_thumbnail_path(self, project_id: UUID) -> Path | None:
        """Get the path to the project's thumbnail image, as determined by the earliest media"""
        media_repo = MediaRepository(project_id=str(project_id), db=self.db_session)
        earliest_media = media_repo.get_earliest()

        if earliest_media:
            return self._projects_dir / f"{project_id}/dataset/{earliest_media.id}-thumb.jpg"
        return None

    def _to_project(self, project_db: ProjectDB) -> Project:
        """Convert database model to domain model with enriched runtime data."""
        project_id = UUID(project_db.id)
        active_pipeline = self._pipeline_service.is_running(project_id=project_id)
        labels = self._label_service.list_all(project_id=project_id)
        return Project.model_validate(
            {
                **project_db.__dict__,
                "active_pipeline": active_pipeline,
                "task": {
                    "task_type": project_db.task_type,
                    "exclusive_labels": project_db.exclusive_labels,
                    "labels": labels,
                },
            }
        )
