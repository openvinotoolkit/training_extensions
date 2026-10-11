# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.models import UploadState


class UploadView(BaseModel):
    """State of a resumable (TUS) upload."""

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": "4f8c9a4e-6d3b-4b0e-9f7a-2c1d5e8b7a90",
                "filename": "grapes.mp4",
                "size": 734003200,
                "offset": 412876800,
                "state": "in_progress",
                "created_at": "2026-09-30T10:00:00Z",
                "expires_at": "2026-10-01T10:05:00Z",
            }
        },
    )

    id: UUID = Field(..., description="Unique identifier of the upload")
    filename: str = Field(
        ..., description="Sanitised filename declared by the client; never used as a path on the server"
    )
    size: int | None = Field(..., description="Total size in bytes; null when the client deferred the length")
    offset: int = Field(..., description="Number of bytes received and acknowledged by the server")
    state: UploadState = Field(..., description="Lifecycle state of the upload")
    created_at: datetime = Field(..., description="Creation time")
    expires_at: datetime = Field(..., description="Time after which the upload is discarded if not completed")


class FromUploadRequest(BaseModel):
    """Request to consume a completed resumable upload."""

    upload_id: UUID = Field(..., description="ID of a completed resumable upload")

    model_config = {"json_schema_extra": {"example": {"upload_id": "4f8c9a4e-6d3b-4b0e-9f7a-2c1d5e8b7a90"}}}


class MediaFromUploadRequest(FromUploadRequest):
    """Request to create a dataset media from a completed resumable upload."""

    name: str | None = Field(
        None,
        min_length=1,
        max_length=255,
        description="Optional media name overriding the uploaded filename (without extension)",
    )

    model_config = {
        "json_schema_extra": {"example": {"upload_id": "4f8c9a4e-6d3b-4b0e-9f7a-2c1d5e8b7a90", "name": "grapes"}}
    }
