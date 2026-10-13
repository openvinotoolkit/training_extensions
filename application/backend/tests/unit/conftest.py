# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Generator, Iterator
from contextlib import contextmanager
from datetime import timedelta
from multiprocessing.synchronize import Condition
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.dependencies import get_project, get_upload_service
from app.api.schemas import LabelView, ProjectView, TaskView
from app.db.schema import UploadDB
from app.main import create_app
from app.models import TaskType
from app.services import MetricsService, UploadService
from app.services.event.event_bus import EventBus


@pytest.fixture(scope="session")
def fxt_app() -> FastAPI:
    app = create_app()
    # The lifespan (which populates the app state) does not run for the test app, so provide the
    # pieces of state that dependencies resolve from it.
    app.state.event_bus = MagicMock(spec=EventBus)
    return app


@pytest.fixture
def fxt_get_project(fxt_app: FastAPI) -> Generator[ProjectView]:
    project = MagicMock(
        spec=ProjectView,
        id=uuid4(),
        task=TaskView(
            task_type=TaskType.CLASSIFICATION,
            exclusive_labels=True,
            labels=[
                LabelView(id=uuid4(), name="cat", color="#11AA22", hotkey="s"),
                LabelView(id=uuid4(), name="dog", color="#AA2233", hotkey="d"),
            ],
        ),
    )
    fxt_app.dependency_overrides[get_project] = lambda: project
    yield project
    del fxt_app.dependency_overrides[get_project]


@pytest.fixture
def fxt_event_bus() -> MagicMock:
    return MagicMock(spec=EventBus)


@pytest.fixture
def fxt_metrics_service() -> MagicMock:
    return MagicMock(spec=MetricsService)


@pytest.fixture
def fxt_condition() -> MagicMock:
    return MagicMock(spec=Condition)


@pytest.fixture
def fxt_upload_service(fxt_app: FastAPI, tmp_path: Path) -> Generator[UploadService]:
    """A real resumable upload service, backed by an in-memory database and injected in the app."""
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    UploadDB.metadata.create_all(engine, tables=[UploadDB.__table__])  # pyrefly: ignore[bad-argument-type]
    session_maker = sessionmaker(bind=engine)

    @contextmanager
    def session_factory() -> Iterator[Session]:
        with session_maker() as session, session.begin():
            yield session

    service = UploadService(
        uploads_dir=tmp_path / "uploads",
        max_size=1024,
        ttl=timedelta(hours=1),
        session_factory=session_factory,
    )
    fxt_app.dependency_overrides[get_upload_service] = lambda: service
    yield service
    fxt_app.dependency_overrides.pop(get_upload_service, None)
    engine.dispose()
