# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from loguru import logger
from pydantic import Field, computed_field

from app.core.jobs.models import JobParams, JobType, ProjectJob
from app.models.project import Task
from app.models.system import DeviceInfo


class TrainingJobParams(JobParams):
    job_id: UUID
    project_id: UUID
    model_architecture_id: str
    model_architecture_name: str
    parent_model_revision_id: UUID | None = None
    task: Task
    model_id: UUID = Field(default_factory=uuid4)
    dataset_revision_id: UUID | None = None
    device: DeviceInfo

    @computed_field  # type: ignore[prop-decorator]
    @property
    def has_model_revision(self) -> bool:
        return self.parent_model_revision_id is not None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def model_name(self) -> str:
        """User-friendly model name derived from architecture name and model ID."""
        return f"{self.model_architecture_name} ({str(self.model_id).split('-')[0]})"


class TrainingJob(ProjectJob[TrainingJobParams]):
    job_type: Literal[JobType.TRAIN] = JobType.TRAIN  # pyrefly: ignore[bad-override]
    log_dir: Path
    data_dir: Path
    params: TrainingJobParams

    # TODO (see github.com/open-edge-platform/geti/pull/7494#discussion_r3935273273)
    # Consider refactoring to avoid placing this logic in the model layer: one possible
    # approach is to rely on domain events (the model publishes events, a decoupled
    # listener handles them); another approach is to let the controller handle the
    # completion logic based on the job type.
    def on_complete(self) -> None:
        """Copy the training log and clean up the getitune workspace upon job completion."""
        log_path = self.log_dir / self.log_file
        if not log_path.exists():
            logger.warning(f"Log file {log_path} does not exist")
        else:
            new_path = (
                self.data_dir
                / "projects"
                / str(self.project_id)
                / "models"
                / str(self.params.model_id)
                / "training.log"
            )
            new_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(log_path, new_path)

        # Remove the getitune workspace directory (parent of the timestamped subdir)
        # so it never lingers on disk regardless of whether the job succeeded or failed.
        workspace_dir = self.data_dir / f"getitune-workspace-{self.params.model_id}"
        try:
            shutil.rmtree(workspace_dir)
            logger.info(f"Cleaned up getitune workspace directory at {workspace_dir}")
        except FileNotFoundError:
            # Directory was never created or already removed; treat as a no-op.
            pass
        except Exception as cleanup_exc:
            logger.error(f"Failed to clean up getitune workspace directory at {workspace_dir}: {cleanup_exc}")

        try:
            self._reconcile_stuck_training_status()
        except Exception as reconcile_exc:
            logger.error(f"Failed to verify the training status for model {self.params.model_id}: {reconcile_exc}")

    def _reconcile_stuck_training_status(self) -> None:
        """Force training_status to FAILED if the trainer process died without updating it (e.g. a crash/kill)."""
        # Imported locally: the "models" layer normally must not depend on the "services" layer;
        # this is a deliberate, narrow exception to provide a safety net at job completion time.
        from app.db.engine import get_db_session
        from app.models import TrainingStatus
        from app.services import ModelService, ResourceNotFoundError

        with get_db_session() as db:
            model_service = ModelService(data_dir=self.data_dir, db_session=db)
            try:
                model = model_service.get_model(project_id=self.project_id, model_id=self.params.model_id)
            except ResourceNotFoundError:
                return  # revision was never created (or was already deleted), nothing to reconcile

            if model.training_info is not None:
                found_status = model.training_info.status
                if found_status not in (TrainingStatus.SUCCESSFUL, TrainingStatus.FAILED):
                    model_service.update_revision_status(
                        project_id=self.project_id,
                        model_id=self.params.model_id,
                        training_status=TrainingStatus.FAILED,
                        training_finished_at=datetime.now(UTC),
                    )
                    logger.warning(
                        f"Model {self.params.model_id} was left at {found_status} after job completion; "
                        "forced to FAILED"
                    )
