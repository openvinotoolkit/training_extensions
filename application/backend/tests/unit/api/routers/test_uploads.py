# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import base64
import hashlib
from uuid import uuid4

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from app.services import UploadService

TUS = {"Tus-Resumable": "1.0.0"}
OCTET_STREAM = {"Content-Type": "application/offset+octet-stream"}


def _metadata(**values: str) -> str:
    return ",".join(f"{key} {base64.b64encode(value.encode()).decode()}" for key, value in values.items())


def _create(client: TestClient, length: int, data: bytes | None = None, **headers: str) -> str:
    request_headers = {**TUS, "Upload-Length": str(length), **headers}
    if data is not None:
        request_headers.update(OCTET_STREAM)
    response = client.post("/api/uploads", headers=request_headers, content=data)
    assert response.status_code == status.HTTP_201_CREATED, response.text
    return response.headers["Location"]


def _patch(client: TestClient, location: str, offset: int, data: bytes, **headers: str):
    return client.patch(
        location, headers={**TUS, **OCTET_STREAM, "Upload-Offset": str(offset), **headers}, content=data
    )


@pytest.mark.usefixtures("fxt_upload_service")
class TestUploadEndpoints:
    def test_options(self, fxt_client: TestClient, fxt_upload_service: UploadService) -> None:
        response = fxt_client.options("/api/uploads")

        assert response.status_code == status.HTTP_204_NO_CONTENT
        assert response.headers["Tus-Resumable"] == "1.0.0"
        assert response.headers["Tus-Version"] == "1.0.0"
        extensions = response.headers["Tus-Extension"].split(",")
        assert {"creation", "creation-with-upload", "termination", "expiration", "checksum"} <= set(extensions)
        assert response.headers["Tus-Max-Size"] == str(fxt_upload_service.max_size)
        assert "sha1" in response.headers["Tus-Checksum-Algorithm"].split(",")

    def test_create(self, fxt_client: TestClient) -> None:
        response = fxt_client.post(
            "/api/uploads",
            headers={**TUS, "Upload-Length": "10", "Upload-Metadata": _metadata(filename="grapes.mp4")},
        )

        assert response.status_code == status.HTTP_201_CREATED
        location = response.headers["Location"]
        assert location.startswith("/api/uploads/")
        assert response.headers["Upload-Offset"] == "0"
        assert response.headers["Upload-Length"] == "10"
        assert response.headers["Upload-Expires"].endswith("GMT")
        assert response.headers["Tus-Resumable"] == "1.0.0"

        view = fxt_client.get(location).json()
        assert view["filename"] == "grapes.mp4"
        assert view["size"] == 10
        assert view["offset"] == 0
        assert view["state"] == "pending"
        assert {"id", "created_at", "expires_at"} <= view.keys()

    def test_create_with_upload(self, fxt_client: TestClient) -> None:
        response = fxt_client.post(
            "/api/uploads", headers={**TUS, **OCTET_STREAM, "Upload-Length": "6"}, content=b"abc"
        )

        assert response.status_code == status.HTTP_201_CREATED
        assert response.headers["Upload-Offset"] == "3"

    def test_create_with_deferred_length(self, fxt_client: TestClient) -> None:
        response = fxt_client.post("/api/uploads", headers={**TUS, "Upload-Defer-Length": "1"})

        assert response.status_code == status.HTTP_201_CREATED
        assert response.headers["Upload-Defer-Length"] == "1"
        assert "Upload-Length" not in response.headers
        assert fxt_client.get(response.headers["Location"]).json()["size"] is None

    @pytest.mark.parametrize(
        "headers",
        [
            {},
            {"Upload-Length": "-1"},
            {"Upload-Length": "abc"},
            {"Upload-Length": "1", "Upload-Defer-Length": "1"},
            {"Upload-Defer-Length": "2"},
            {"Upload-Length": "1", "Upload-Metadata": "filename not-base64!"},
        ],
    )
    def test_create_bad_request(self, fxt_client: TestClient, headers: dict[str, str]) -> None:
        response = fxt_client.post("/api/uploads", headers={**TUS, **headers})

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert response.headers["Tus-Resumable"] == "1.0.0"

    def test_create_too_large(self, fxt_client: TestClient, fxt_upload_service: UploadService) -> None:
        response = fxt_client.post(
            "/api/uploads", headers={**TUS, "Upload-Length": str(fxt_upload_service.max_size + 1)}
        )

        assert response.status_code == status.HTTP_413_CONTENT_TOO_LARGE
        assert fxt_client.get("/api/uploads").json() == []

    def test_create_unsupported_version(self, fxt_client: TestClient) -> None:
        response = fxt_client.post("/api/uploads", headers={"Tus-Resumable": "0.2.0", "Upload-Length": "1"})

        assert response.status_code == status.HTTP_412_PRECONDITION_FAILED
        assert response.headers["Tus-Version"] == "1.0.0"

    def test_create_with_body_of_wrong_type(self, fxt_client: TestClient) -> None:
        response = fxt_client.post(
            "/api/uploads",
            headers={**TUS, "Upload-Length": "3", "Content-Type": "application/octet-stream"},
            content=b"abc",
        )

        assert response.status_code == status.HTTP_415_UNSUPPORTED_MEDIA_TYPE

    def test_method_override_is_rejected(self, fxt_client: TestClient) -> None:
        location = _create(fxt_client, 3)

        response = fxt_client.post(location, headers={**TUS, "X-HTTP-Method-Override": "DELETE"})
        assert response.status_code in (status.HTTP_400_BAD_REQUEST, status.HTTP_405_METHOD_NOT_ALLOWED)
        response = fxt_client.post(
            "/api/uploads", headers={**TUS, "Upload-Length": "1", "X-HTTP-Method-Override": "PATCH"}
        )
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert fxt_client.get(location).status_code == status.HTTP_200_OK

    def test_patch_and_resume(self, fxt_client: TestClient) -> None:
        location = _create(fxt_client, 6)

        response = _patch(fxt_client, location, 0, b"abc")
        assert response.status_code == status.HTTP_204_NO_CONTENT
        assert response.headers["Upload-Offset"] == "3"
        assert response.headers["Tus-Resumable"] == "1.0.0"

        head = fxt_client.head(location, headers=TUS)
        assert head.status_code == status.HTTP_204_NO_CONTENT
        assert head.headers["Upload-Offset"] == "3"
        assert head.headers["Upload-Length"] == "6"
        assert head.headers["Cache-Control"] == "no-store"

        response = _patch(fxt_client, location, 3, b"def")
        assert response.status_code == status.HTTP_204_NO_CONTENT
        assert response.headers["Upload-Offset"] == "6"
        assert fxt_client.get(location).json()["state"] == "completed"

    def test_patch_offset_mismatch(self, fxt_client: TestClient) -> None:
        location = _create(fxt_client, 6, b"abc")

        response = _patch(fxt_client, location, 0, b"abc")

        assert response.status_code == status.HTTP_409_CONFLICT

    def test_patch_past_upload_length(self, fxt_client: TestClient) -> None:
        location = _create(fxt_client, 3)

        response = _patch(fxt_client, location, 0, b"abcd")

        assert response.status_code == status.HTTP_413_CONTENT_TOO_LARGE
        assert fxt_client.head(location, headers=TUS).headers["Upload-Offset"] == "0"

    def test_patch_wrong_content_type(self, fxt_client: TestClient) -> None:
        location = _create(fxt_client, 3)

        response = fxt_client.patch(
            location, headers={**TUS, "Upload-Offset": "0", "Content-Type": "application/json"}, content=b"abc"
        )

        assert response.status_code == status.HTTP_415_UNSUPPORTED_MEDIA_TYPE

    @pytest.mark.parametrize("offset", [None, "-1", "x"])
    def test_patch_invalid_offset(self, fxt_client: TestClient, offset: str | None) -> None:
        location = _create(fxt_client, 3)
        headers = {**TUS, **OCTET_STREAM}
        if offset is not None:
            headers["Upload-Offset"] = offset

        response = fxt_client.patch(location, headers=headers, content=b"abc")

        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_patch_checksum(self, fxt_client: TestClient) -> None:
        location = _create(fxt_client, 6)
        good = "sha1 " + base64.b64encode(hashlib.sha1(b"abc").digest()).decode()
        bad = "sha1 " + base64.b64encode(hashlib.sha1(b"xyz").digest()).decode()

        assert _patch(fxt_client, location, 0, b"abc", **{"Upload-Checksum": good}).status_code == 204
        response = _patch(fxt_client, location, 3, b"def", **{"Upload-Checksum": bad})

        assert response.status_code == 460
        assert fxt_client.head(location, headers=TUS).headers["Upload-Offset"] == "3"

    def test_patch_deferred_length(self, fxt_client: TestClient) -> None:
        response = fxt_client.post("/api/uploads", headers={**TUS, "Upload-Defer-Length": "1"})
        location = response.headers["Location"]

        response = _patch(fxt_client, location, 0, b"abc", **{"Upload-Length": "3"})

        assert response.status_code == status.HTTP_204_NO_CONTENT
        assert response.headers["Upload-Length"] == "3"
        assert fxt_client.get(location).json()["state"] == "completed"

    def test_delete(self, fxt_client: TestClient) -> None:
        location = _create(fxt_client, 6, b"abc")

        response = fxt_client.delete(location, headers=TUS)

        assert response.status_code == status.HTTP_204_NO_CONTENT
        assert fxt_client.get(location).status_code == status.HTTP_404_NOT_FOUND
        assert fxt_client.head(location, headers=TUS).status_code == status.HTTP_404_NOT_FOUND

    def test_not_found(self, fxt_client: TestClient) -> None:
        location = f"/api/uploads/{uuid4()}"

        assert fxt_client.head(location, headers=TUS).status_code == status.HTTP_404_NOT_FOUND
        assert _patch(fxt_client, location, 0, b"a").status_code == status.HTTP_404_NOT_FOUND
        assert fxt_client.delete(location, headers=TUS).status_code == status.HTTP_404_NOT_FOUND
        assert fxt_client.get(location).status_code == status.HTTP_404_NOT_FOUND

    def test_list(self, fxt_client: TestClient) -> None:
        first = _create(fxt_client, 3)
        second = _create(fxt_client, 3)

        response = fxt_client.get("/api/uploads")

        assert response.status_code == status.HTTP_200_OK
        assert [f"/api/uploads/{upload['id']}" for upload in response.json()] == [second, first]
