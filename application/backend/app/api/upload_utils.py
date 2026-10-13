# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Helpers shared by the endpoints that create or consume resumable (TUS) uploads."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID

from fastapi import HTTPException, status

from app.services.upload_service import (
    ClaimedUpload,
    UploadChecksumMismatchError,
    UploadError,
    UploadGoneError,
    UploadInvalidError,
    UploadLockedError,
    UploadNotFoundError,
    UploadNotReadyError,
    UploadOffsetMismatchError,
    UploadService,
    UploadTooLargeError,
)

# Non-standard status code defined by the TUS checksum extension.
HTTP_460_CHECKSUM_MISMATCH = 460

_STATUS_CODES: list[tuple[type[UploadError], int]] = [
    (UploadNotFoundError, status.HTTP_404_NOT_FOUND),
    (UploadGoneError, status.HTTP_410_GONE),
    (UploadInvalidError, status.HTTP_400_BAD_REQUEST),
    (UploadTooLargeError, status.HTTP_413_CONTENT_TOO_LARGE),
    (UploadOffsetMismatchError, status.HTTP_409_CONFLICT),
    (UploadNotReadyError, status.HTTP_409_CONFLICT),
    (UploadLockedError, status.HTTP_423_LOCKED),
    (UploadChecksumMismatchError, HTTP_460_CHECKSUM_MISMATCH),
]

# OpenAPI documentation of the errors returned when consuming an upload.
FROM_UPLOAD_RESPONSES: dict[int | str, dict] = {
    status.HTTP_404_NOT_FOUND: {"description": "Upload not found"},
    status.HTTP_409_CONFLICT: {"description": "Upload is not complete yet"},
    status.HTTP_410_GONE: {"description": "Upload has expired or has already been consumed"},
    status.HTTP_422_UNPROCESSABLE_CONTENT: {"description": "Uploaded file is invalid or has an unsupported format"},
    status.HTTP_423_LOCKED: {"description": "Another request is currently operating on the upload"},
}


def upload_error_to_http(error: UploadError, headers: dict[str, str] | None = None) -> HTTPException:
    """Translate an upload service error to the matching HTTP error."""
    status_code = next(
        (code for error_type, code in _STATUS_CODES if isinstance(error, error_type)),
        status.HTTP_400_BAD_REQUEST,
    )
    return HTTPException(status_code=status_code, detail=str(error), headers=headers)


@asynccontextmanager
async def consume_upload(upload_service: UploadService, upload_id: UUID) -> AsyncIterator[ClaimedUpload]:
    """
    Claim a completed upload for the duration of the block, translating claim errors to HTTP errors.

    If the block raises, the upload is released and can be consumed again (or deleted by the client).
    """
    try:
        async with upload_service.consume(upload_id) as claimed:
            yield claimed
    except UploadError as error:
        raise upload_error_to_http(error) from error


def split_upload_filename(filename: str) -> tuple[str, str]:
    """
    Split the sanitised filename of an upload into its name and extension (without the leading dot).

    Raises:
        HTTPException: 422 if the filename has no name or no extension.
    """
    name, dot, extension = filename.rpartition(".")
    name = name.strip()
    if not dot or not name or not extension:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"The uploaded filename '{filename}' must have a name and an extension.",
        )
    return name, extension
