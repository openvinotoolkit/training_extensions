# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Generator
from datetime import datetime
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi import FastAPI, status

from app.api.dependencies import get_dataset_view, get_dataset_view_service
from app.models.dataset_view import DatasetView
from app.services import DatasetViewService


@pytest.fixture
def fxt_dataset_view_service(fxt_app: FastAPI) -> Generator[MagicMock]:
    dataset_view_service = MagicMock(spec=DatasetViewService)
    fxt_app.dependency_overrides[get_dataset_view_service] = lambda: dataset_view_service
    yield dataset_view_service
    fxt_app.dependency_overrides.pop(get_dataset_view_service, None)


@pytest.fixture
def fxt_dataset_view(fxt_get_project) -> DatasetView:
    return DatasetView(
        id=uuid4(),
        project_id=fxt_get_project.id,
        name="Canada signs",
        created_at=datetime(2026, 1, 1),
    )


@pytest.fixture
def fxt_get_dataset_view(fxt_app: FastAPI, fxt_dataset_view: DatasetView):
    fxt_app.dependency_overrides[get_dataset_view] = lambda: fxt_dataset_view
    yield fxt_dataset_view
    fxt_app.dependency_overrides.pop(get_dataset_view, None)


class TestDatasetViewEndpoints:
    def test_create_dataset_view_success(self, fxt_get_project, fxt_dataset_view_service, fxt_client, fxt_dataset_view):
        fxt_dataset_view_service.create_dataset_view.return_value = fxt_dataset_view

        response = fxt_client.post(
            f"/api/projects/{fxt_get_project.id}/dataset/views",
            json={"name": "Canada signs"},
        )

        assert response.status_code == status.HTTP_201_CREATED
        assert response.json()["name"] == "Canada signs"
        fxt_dataset_view_service.create_dataset_view.assert_called_once_with(
            project_id=fxt_get_project.id, name="Canada signs", media_ids=None
        )

    def test_create_dataset_view_not_implemented(self, fxt_get_project, fxt_dataset_view_service, fxt_client):
        fxt_dataset_view_service.create_dataset_view.side_effect = NotImplementedError

        response = fxt_client.post(
            f"/api/projects/{fxt_get_project.id}/dataset/views",
            json={"name": "Canada signs"},
        )

        assert response.status_code == status.HTTP_501_NOT_IMPLEMENTED

    def test_list_dataset_views_success(self, fxt_get_project, fxt_dataset_view_service, fxt_client, fxt_dataset_view):
        fxt_dataset_view_service.list_dataset_views.return_value = [fxt_dataset_view]

        response = fxt_client.get(f"/api/projects/{fxt_get_project.id}/dataset/views")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)
        assert data[0]["id"] == str(fxt_dataset_view.id)
        fxt_dataset_view_service.list_dataset_views.assert_called_once_with(project_id=fxt_get_project.id)

    def test_get_dataset_view_details_success(self, fxt_get_project, fxt_client, fxt_get_dataset_view):
        response = fxt_client.get(f"/api/projects/{fxt_get_project.id}/dataset/views/{fxt_get_dataset_view.id}")

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["name"] == "Canada signs"

    def test_get_dataset_view_details_not_implemented(self, fxt_get_project, fxt_dataset_view_service, fxt_client):
        """When the underlying service logic isn't implemented yet, the API returns 501."""
        fxt_dataset_view_service.get_dataset_view_by_id.side_effect = NotImplementedError

        response = fxt_client.get(f"/api/projects/{fxt_get_project.id}/dataset/views/{uuid4()}")

        assert response.status_code == status.HTTP_501_NOT_IMPLEMENTED

    def test_rename_dataset_view_success(
        self, fxt_get_project, fxt_get_dataset_view, fxt_dataset_view_service, fxt_client
    ):
        renamed = fxt_get_dataset_view.model_copy(update={"name": "New name"})
        fxt_dataset_view_service.rename_dataset_view.return_value = renamed

        response = fxt_client.patch(
            f"/api/projects/{fxt_get_project.id}/dataset/views/{fxt_get_dataset_view.id}",
            json={"name": "New name"},
        )

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["name"] == "New name"
        fxt_dataset_view_service.rename_dataset_view.assert_called_once_with(
            project_id=fxt_get_project.id, dataset_view_id=fxt_get_dataset_view.id, new_name="New name"
        )

    def test_delete_dataset_view_success(
        self, fxt_get_project, fxt_get_dataset_view, fxt_dataset_view_service, fxt_client
    ):
        response = fxt_client.delete(f"/api/projects/{fxt_get_project.id}/dataset/views/{fxt_get_dataset_view.id}")

        assert response.status_code == status.HTTP_204_NO_CONTENT
        fxt_dataset_view_service.delete_dataset_view.assert_called_once_with(
            project_id=fxt_get_project.id, dataset_view_id=fxt_get_dataset_view.id
        )

    def test_assign_media_to_dataset_view_success(
        self, fxt_get_project, fxt_get_dataset_view, fxt_dataset_view_service, fxt_client
    ):
        media_id = uuid4()

        response = fxt_client.post(
            f"/api/projects/{fxt_get_project.id}/dataset/views/{fxt_get_dataset_view.id}/media",
            json={"media_ids": [str(media_id)]},
        )

        assert response.status_code == status.HTTP_204_NO_CONTENT
        fxt_dataset_view_service.assign_media.assert_called_once_with(
            project_id=fxt_get_project.id, dataset_view_id=fxt_get_dataset_view.id, media_ids=[media_id]
        )

    def test_unassign_media_from_dataset_view_success(
        self, fxt_get_project, fxt_get_dataset_view, fxt_dataset_view_service, fxt_client
    ):
        media_id = uuid4()

        # httpx's `.delete()` helper doesn't support a request body, so a DELETE with a JSON payload
        # must be issued via the generic `.request()` method.
        response = fxt_client.request(
            "DELETE",
            f"/api/projects/{fxt_get_project.id}/dataset/views/{fxt_get_dataset_view.id}/media",
            json={"media_ids": [str(media_id)]},
        )

        assert response.status_code == status.HTTP_204_NO_CONTENT
        fxt_dataset_view_service.unassign_media.assert_called_once_with(
            project_id=fxt_get_project.id, dataset_view_id=fxt_get_dataset_view.id, media_ids=[media_id]
        )

    def test_list_dataset_view_media_endpoint_removed(
        self, fxt_get_project, fxt_get_dataset_view, fxt_dataset_view_service, fxt_client
    ):
        """Listing the media of a view is only served by `GET /dataset/media?dataset_view_id=...`."""
        response = fxt_client.get(
            f"/api/projects/{fxt_get_project.id}/dataset/views/{fxt_get_dataset_view.id}/media",
            params={"limit": 5, "offset": 0},
        )

        assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED
        fxt_dataset_view_service.list_dataset_view_media.assert_not_called()

    def test_rename_dataset_view_invalid_id(self, fxt_get_project, fxt_dataset_view_service, fxt_client):
        response = fxt_client.patch(
            f"/api/projects/{fxt_get_project.id}/dataset/views/invalid-id",
            json={"name": "New name"},
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        fxt_dataset_view_service.rename_dataset_view.assert_not_called()
