# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import cast

from sqlalchemy import ColumnElement, CursorResult, delete, or_, select, update
from sqlalchemy.orm import Session

from app.db.schema import UploadDB
from app.models import UploadState
from app.repositories.base import BaseRepository


def _is_collectable(now: datetime) -> ColumnElement[bool]:
    return or_(UploadDB.expires_at < now, UploadDB.state == UploadState.CONSUMED)


class UploadRepository(BaseRepository[UploadDB]):
    """Repository for resumable upload metadata."""

    def __init__(self, db: Session):
        super().__init__(db, UploadDB)

    def list_all(self) -> Sequence[UploadDB]:
        """List all uploads, most recent first."""
        return self.db.execute(select(UploadDB).order_by(UploadDB.created_at.desc())).scalars().all()

    def list_collectable(self, now: datetime) -> Sequence[UploadDB]:
        """List uploads that can be garbage-collected: expired or already consumed."""
        stmt = select(UploadDB).where(_is_collectable(now))
        return self.db.execute(stmt).scalars().all()

    def delete_if_collectable(self, upload_id: str, now: datetime) -> bool:
        """
        Delete an upload with a single guarded DELETE, only if it is still expired or consumed.

        Returns:
            True if the upload was deleted, False otherwise.
        """
        stmt = delete(UploadDB).where(UploadDB.id == upload_id, _is_collectable(now))
        result = cast(CursorResult, self.db.execute(stmt))
        return result.rowcount == 1

    def transition_state(self, upload_id: str, from_state: UploadState, to_state: UploadState) -> bool:
        """
        Atomically move an upload from one state to another with a single guarded UPDATE.

        Returns:
            True if the upload was in `from_state` and has been updated, False otherwise.
        """
        stmt = (
            update(UploadDB)
            .where(UploadDB.id == upload_id, UploadDB.state == from_state)
            .values(state=to_state, updated_at=datetime.now(UTC))
        )
        result = cast(CursorResult, self.db.execute(stmt))
        return result.rowcount == 1

    def claim(self, upload_id: str, now: datetime) -> bool:
        """
        Atomically mark a completed, non-expired upload as consumed.

        This single guarded UPDATE guarantees that one upload can be consumed at most once.

        Returns:
            True if the upload was claimed by this call, False otherwise.
        """
        stmt = (
            update(UploadDB)
            .where(
                UploadDB.id == upload_id,
                UploadDB.state == UploadState.COMPLETED,
                UploadDB.expires_at >= now,
            )
            .values(state=UploadState.CONSUMED, updated_at=now)
        )
        result = cast(CursorResult, self.db.execute(stmt))
        return result.rowcount == 1
