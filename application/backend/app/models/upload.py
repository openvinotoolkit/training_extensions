# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from enum import StrEnum

from pydantic import ConfigDict

from app.core.models import BaseRequiredIDModel


class UploadState(StrEnum):
    """Lifecycle state of a resumable upload."""

    PENDING = "pending"  # created, no bytes received yet
    IN_PROGRESS = "in_progress"  # some bytes received
    COMPLETED = "completed"  # all declared bytes received, ready to be consumed
    CONSUMED = "consumed"  # moved into its final destination (media, staged dataset, ...)


class Upload(BaseRequiredIDModel):
    """A resumable (TUS) upload tracked by the server."""

    model_config = ConfigDict(from_attributes=True)

    filename: str
    content_type: str | None = None
    size: int | None = None  # None when the client deferred the length (Upload-Defer-Length)
    offset: int = 0
    state: UploadState
    checksum: str | None = None
    created_at: datetime
    updated_at: datetime
    expires_at: datetime

    @property
    def is_complete(self) -> bool:
        """Whether all the declared bytes have been received."""
        return self.size is not None and self.offset == self.size
