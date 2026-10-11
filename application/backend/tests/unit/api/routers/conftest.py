# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import base64
from collections.abc import AsyncGenerator, Callable
from unittest.mock import Mock
from uuid import UUID

import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from app.api.dependencies import get_dataset_view_service, get_project_service
from app.services import DatasetViewService, ProjectService, UploadService


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


@pytest.fixture
def fxt_create_upload(fxt_client: TestClient, fxt_upload_service: UploadService) -> Callable[..., UUID]:
    """Create a resumable upload through the TUS API; it is complete unless a larger `length` is given."""

    def create(filename: str, data: bytes = b"data", length: int | None = None) -> UUID:
        encoded_filename = base64.b64encode(filename.encode()).decode()
        response = fxt_client.post(
            "/api/uploads",
            headers={
                "Tus-Resumable": "1.0.0",
                "Upload-Length": str(length if length is not None else len(data)),
                "Upload-Metadata": f"filename {encoded_filename}",
                "Content-Type": "application/offset+octet-stream",
            },
            content=data,
        )
        assert response.status_code == 201, response.text
        return UUID(response.headers["Location"].rsplit("/", 1)[-1])

    return create
