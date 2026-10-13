# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Generator
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from app.services import (
    DatasetService,
    LabelService,
    MediaService,
    ModelService,
    PipelineService,
    ProjectService,
    SystemService,
)
from app.services.dataset_view_service import DatasetViewService
from app.services.event.event_bus import EventBus
from app.services.video import VideoService


@pytest.fixture
def fxt_projects_dir(tmp_path: Path) -> Generator[Path]:
    """Set up a temporary data directory for tests."""
    projects_dir = Path(tmp_path / "projects")
    projects_dir.mkdir(parents=True, exist_ok=True)
    yield projects_dir


@pytest.fixture
def fxt_system_service() -> SystemService:
    """Fixture to create a SystemService instance."""
    return SystemService()


@pytest.fixture
def fxt_pipeline_service(
    fxt_event_bus: EventBus,
    db_session: Session,
    fxt_system_service: SystemService,
    fxt_projects_dir: Path,
) -> PipelineService:
    """Fixture to create a PipelineService instance."""
    return PipelineService(
        event_bus=fxt_event_bus,
        db_session=db_session,
        system_service=fxt_system_service,
        model_service=ModelService(data_dir=fxt_projects_dir.parent, db_session=db_session),
    )


@pytest.fixture
def fxt_label_service(db_session: Session, fxt_event_bus: EventBus) -> LabelService:
    """Fixture to create a LabelService instance."""
    return LabelService(db_session=db_session, event_bus=fxt_event_bus)


@pytest.fixture
def fxt_project_service(
    fxt_projects_dir: Path, db_session: Session, fxt_pipeline_service: PipelineService, fxt_label_service: LabelService
) -> ProjectService:
    """Fixture to create a ProjectService instance."""
    return ProjectService(
        fxt_projects_dir.parent,
        db_session=db_session,
        pipeline_service=fxt_pipeline_service,
        label_service=fxt_label_service,
    )


@pytest.fixture
def fxt_media_service(
    fxt_projects_dir: Path,
    db_session: Session,
) -> MediaService:
    """Fixture to create a MediaService instance."""
    return MediaService(data_dir=fxt_projects_dir.parent, db_session=db_session, video_service=VideoService())


@pytest.fixture
def fxt_dataset_service(
    fxt_label_service: LabelService,
    fxt_media_service: MediaService,
    db_session: Session,
) -> DatasetService:
    """Fixture to create a DatasetService instance."""
    return DatasetService(label_service=fxt_label_service, media_service=fxt_media_service, db_session=db_session)


@pytest.fixture
def fxt_dataset_view_service(
    db_session: Session,
) -> DatasetViewService:
    """Fixture to create a DatasetViewService instance."""
    return DatasetViewService(db_session=db_session)
