# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from collections.abc import AsyncGenerator
from unittest.mock import Mock

import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from app.api.dependencies import get_dataset_view_service, get_project_service
from app.services import DatasetViewService, ProjectService


@pytest.fixture
def fxt_client(fxt_app: FastAPI) -> TestClient:
    return TestClient(fxt_app)


@pytest.fixture
def fxt_project_service(fxt_app: FastAPI) -> Mock:
    project_service = Mock(spec=ProjectService)
    fxt_app.dependency_overrides[get_project_service] = lambda: project_service
    return project_service


@pytest.fixture
def fxt_dataset_view_service(fxt_app: FastAPI) -> Mock:
    dataset_view_service = Mock(spec=DatasetViewService)
    fxt_app.dependency_overrides[get_dataset_view_service] = lambda: dataset_view_service
    return dataset_view_service


@pytest_asyncio.fixture
async def fxt_async_client(fxt_app: FastAPI) -> AsyncGenerator[AsyncClient]:
    async with AsyncClient(transport=ASGITransport(app=fxt_app), base_url="http://test") as client:
        yield client
