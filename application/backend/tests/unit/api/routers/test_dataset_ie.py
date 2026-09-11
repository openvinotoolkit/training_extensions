# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import io
from pathlib import Path
from unittest.mock import Mock
from uuid import uuid4

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from app.api.dependencies import get_staged_dataset_service
from app.models import DatasetFormat, StagedDataset
from app.services import StagedDatasetService


@pytest.fixture
def fxt_staged_dataset(tmp_path: Path) -> StagedDataset:
    return StagedDataset(
        id=uuid4(),
        filename=str(tmp_path / "dataset-coco.zip"),
        compressed=True,
        format=DatasetFormat.GETI,
        size=2048,
    )


@pytest.fixture
def fxt_staged_dataset_service(fxt_app) -> Mock:
    dataset_service = Mock(spec=StagedDatasetService)
    fxt_app.dependency_overrides[get_staged_dataset_service] = lambda: dataset_service
    return dataset_service


class TestDatasetIEEndpoints:
    def test_upload_dataset_archive_success(
        self, fxt_staged_dataset_service: Mock, fxt_staged_dataset: StagedDataset, fxt_client: TestClient
    ) -> None:
        fxt_staged_dataset_service.upload.return_value = fxt_staged_dataset
        file_content = b"some-bytes"
        files = {"file": ("dataset-coco.zip", io.BytesIO(file_content), "application/zip")}

        response = fxt_client.post("/api/staged_datasets", files=files)

        assert response.status_code == status.HTTP_201_CREATED
        fxt_staged_dataset_service.upload.assert_called_once()
        assert fxt_staged_dataset_service.upload.call_args.kwargs["filename"] == "dataset.zip"
        response_data = response.json()
        assert response_data["id"] == str(fxt_staged_dataset.id)
        assert response_data["compressed"] == fxt_staged_dataset.compressed
        assert response_data["format"] == fxt_staged_dataset.format.value
        assert response_data["ready_for_export"]
        assert not response_data["ready_for_import"]
        assert response_data["size"] == fxt_staged_dataset.size

    def test_upload_dataset_archive_missing_filename(self, fxt_staged_dataset_service: Mock, fxt_client: TestClient):
        files = {"file": ("", io.BytesIO(b"data"), "application/zip")}

        response = fxt_client.post("/api/staged_datasets", files=files)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
        fxt_staged_dataset_service.upload.assert_not_called()

    def test_list_datasets_success(
        self, fxt_staged_dataset_service: Mock, fxt_staged_dataset: StagedDataset, fxt_client: TestClient
    ) -> None:
        fxt_staged_dataset_service.list_all.return_value = [fxt_staged_dataset] * 3

        response = fxt_client.get("/api/staged_datasets")

        assert response.status_code == status.HTTP_200_OK
        fxt_staged_dataset_service.list_all.assert_called_once()
        response_data = response.json()
        assert len(response_data) == 3

    def test_list_datasets_empty(
        self, fxt_staged_dataset_service: Mock, fxt_staged_dataset: StagedDataset, fxt_client: TestClient
    ) -> None:
        fxt_staged_dataset_service.list_all.return_value = []

        response = fxt_client.get("/api/staged_datasets")

        assert response.status_code == status.HTTP_200_OK
        fxt_staged_dataset_service.list_all.assert_called_once()
        response_data = response.json()
        assert len(response_data) == 0

    def test_get_dataset_success(
        self, fxt_staged_dataset_service: Mock, fxt_staged_dataset: StagedDataset, fxt_client: TestClient
    ) -> None:
        fxt_staged_dataset_service.find_by_id.return_value = fxt_staged_dataset

        response = fxt_client.get(f"/api/staged_datasets/{fxt_staged_dataset.id}")

        assert response.status_code == status.HTTP_200_OK
        fxt_staged_dataset_service.find_by_id.assert_called_once_with(fxt_staged_dataset.id)
        response_data = response.json()
        assert response_data["id"] == str(fxt_staged_dataset.id)

    def test_get_dataset_not_found(
        self, fxt_staged_dataset_service: Mock, fxt_staged_dataset: StagedDataset, fxt_client: TestClient
    ) -> None:
        fxt_staged_dataset_service.find_by_id.return_value = None

        response = fxt_client.get(f"/api/staged_datasets/{fxt_staged_dataset.id}")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_staged_dataset_service.find_by_id.assert_called_once_with(fxt_staged_dataset.id)

    def test_download_archive_success(
        self,
        tmp_path: Path,
        fxt_staged_dataset_service: Mock,
        fxt_staged_dataset: StagedDataset,
        fxt_client: TestClient,
    ):
        from urllib.parse import quote
        
        file_path = tmp_path / "dataset-coco.zip"
        file_content = b"zip-binary-content"
        file_path.write_bytes(file_content)
        fxt_staged_dataset_service.find_by_id.return_value = fxt_staged_dataset

        response = fxt_client.get(f"/api/staged_datasets/{fxt_staged_dataset.id}/zip")

        assert response.status_code == status.HTTP_200_OK
        assert (
            response.headers["content-disposition"] == f"attachment; filename*=utf-8''{quote(Path(fxt_staged_dataset.filename).name)}"
        )
        assert response.headers["content-type"] == "application/zip"
        assert response.content == file_content

    def test_download_archive_not_found(self, fxt_staged_dataset_service: Mock, fxt_client: TestClient):
        fxt_staged_dataset_service.find_by_id.return_value = None
        staged_dataset_id = uuid4()

        response = fxt_client.get(f"/api/staged_datasets/{staged_dataset_id}/zip")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert response.json()["detail"] == f"Staged dataset with ID `{staged_dataset_id}` not found."

    def test_download_archive_not_compressed(
        self, fxt_staged_dataset_service: Mock, fxt_staged_dataset: StagedDataset, fxt_client: TestClient
    ):
        fxt_staged_dataset.compressed = False
        fxt_staged_dataset_service.find_by_id.return_value = fxt_staged_dataset

        response = fxt_client.get(f"/api/staged_datasets/{fxt_staged_dataset.id}/zip")

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
        assert response.json()["detail"] == "Staged dataset is not in zip format ready for download."

    def test_delete_dataset_success(
        self, fxt_staged_dataset_service: Mock, fxt_staged_dataset: StagedDataset, fxt_client: TestClient
    ) -> None:
        fxt_staged_dataset_service.delete_by_id.return_value = True

        response = fxt_client.delete(f"/api/staged_datasets/{fxt_staged_dataset.id}")

        assert response.status_code == status.HTTP_204_NO_CONTENT
        fxt_staged_dataset_service.delete_by_id.assert_called_once_with(fxt_staged_dataset.id)

    def test_delete_dataset_not_found(
        self, fxt_staged_dataset_service: Mock, fxt_staged_dataset: StagedDataset, fxt_client: TestClient
    ) -> None:
        fxt_staged_dataset_service.delete_by_id.return_value = False

        response = fxt_client.delete(f"/api/staged_datasets/{fxt_staged_dataset.id}")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_staged_dataset_service.delete_by_id.assert_called_once_with(fxt_staged_dataset.id)
