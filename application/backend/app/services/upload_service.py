# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Resumable (TUS 1.0.0) upload storage: metadata in SQLite, bytes in flat files named by UUID."""

import base64
import hashlib
import os
import re
import unicodedata
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import AbstractContextManager, asynccontextmanager, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import partial
from io import BufferedRandom
from pathlib import Path
from typing import Protocol
from uuid import UUID, uuid4

import anyio
from anyio import to_thread
from loguru import logger
from sqlalchemy.orm import Session

from app.db import get_db_session
from app.db.schema import UploadDB
from app.models import Upload, UploadState
from app.repositories import UploadRepository

TUS_VERSION = "1.0.0"
TUS_EXTENSIONS = (
    "creation",
    "creation-with-upload",
    "creation-defer-length",
    "termination",
    "expiration",
    "checksum",
)
TUS_CHECKSUM_ALGORITHMS = ("sha1", "sha256", "md5")

_MAX_METADATA_HEADER_LENGTH = 8 * 1024
_MAX_FILENAME_LENGTH = 255
_MAX_EXTENSION_LENGTH = 16
_MAX_CONTENT_TYPE_LENGTH = 255
_DEFAULT_FILENAME = "upload"
_WRITE_BUFFER_SIZE = 1024 * 1024
# Files in the uploads directory without a DB row are removed by the GC only after this grace period,
# so that a file created concurrently with a GC run is never mistaken for an orphan.
_ORPHAN_FILE_GRACE_PERIOD = timedelta(hours=1)
# TUS metadata keys: non-empty, ASCII, no spaces and no commas.
_METADATA_KEY = re.compile(r"^[\x21-\x2b\x2d-\x7e]+$")


class UploadError(Exception):
    """Base error for resumable upload operations."""


class UploadNotFoundError(UploadError):
    """The requested upload does not exist."""


class UploadGoneError(UploadError):
    """The upload has expired or has already been consumed; it no longer accepts operations."""


class UploadInvalidError(UploadError):
    """The request violates the TUS protocol."""


class UploadTooLargeError(UploadError):
    """The upload exceeds the configured maximum size or its declared length."""


class UploadOffsetMismatchError(UploadError):
    """The client offset does not match the offset acknowledged by the server."""

    def __init__(self, expected: int, received: int) -> None:
        super().__init__(f"Upload offset mismatch: expected {expected}, received {received}.")
        self.expected = expected


class UploadLockedError(UploadError):
    """Another request is currently operating on the upload."""


class UploadNotReadyError(UploadError):
    """The upload is not complete and cannot be consumed yet."""


class UploadChecksumMismatchError(UploadError):
    """The checksum of the received bytes does not match the one declared by the client."""


@dataclass(frozen=True)
class ClaimedUpload:
    """A completed upload reserved for exactly one consumer, with the path of its bytes on disk."""

    upload: Upload
    path: Path


def parse_upload_metadata(header: str | None) -> dict[str, str]:
    """
    Decode a TUS 'Upload-Metadata' header ('key base64value,key2 base64value2,...').

    Raises:
        UploadInvalidError: If the header is malformed, too large, or contains invalid base64/UTF-8 values.
    """
    if header is None or not header.strip():
        return {}
    if len(header) > _MAX_METADATA_HEADER_LENGTH:
        raise UploadInvalidError("Upload-Metadata header is too large.")
    parsed: dict[str, str] = {}
    for pair in header.split(","):
        parts = pair.strip().split(" ")
        if len(parts) not in (1, 2) or not _METADATA_KEY.fullmatch(parts[0]):
            raise UploadInvalidError("Upload-Metadata header is malformed.")
        key = parts[0]
        if key in parsed:
            raise UploadInvalidError(f"Upload-Metadata contains a duplicate key '{key}'.")
        try:
            parsed[key] = base64.b64decode(parts[1], validate=True).decode("utf-8") if len(parts) == 2 else ""
        except (ValueError, UnicodeDecodeError) as error:
            raise UploadInvalidError("Upload-Metadata values must be base64-encoded UTF-8 strings.") from error
    return parsed


def parse_checksum_header(header: str) -> tuple[str, bytes]:
    """
    Decode a TUS 'Upload-Checksum' header ('<algorithm> <base64 digest>').

    Raises:
        UploadInvalidError: If the header is malformed or the algorithm is not supported.
    """
    parts = header.strip().split(" ")
    if len(parts) != 2:
        raise UploadInvalidError("Upload-Checksum header is malformed.")
    algorithm = parts[0].lower()
    if algorithm not in TUS_CHECKSUM_ALGORITHMS:
        raise UploadInvalidError(
            f"Unsupported checksum algorithm '{parts[0]}'. Supported: {', '.join(TUS_CHECKSUM_ALGORITHMS)}."
        )
    try:
        digest = base64.b64decode(parts[1], validate=True)
    except ValueError as error:
        raise UploadInvalidError("Upload-Checksum digest must be base64-encoded.") from error
    return algorithm, digest


def sanitize_filename(raw: str | None) -> str:
    """
    Reduce a client-declared filename to a safe, length-capped base name.

    The result is only ever used as a display name and to infer the file extension; it is never used to build
    a filesystem path.
    """
    name = unicodedata.normalize("NFC", raw or "").replace("\\", "/").rsplit("/", 1)[-1]
    name = "".join(char for char in name if char.isprintable()).strip().lstrip(".").strip()
    if not name:
        return _DEFAULT_FILENAME
    if len(name) > _MAX_FILENAME_LENGTH:
        stem, dot, extension = name.rpartition(".")
        if dot and stem and len(extension) <= _MAX_EXTENSION_LENGTH:
            name = f"{stem[: _MAX_FILENAME_LENGTH - len(extension) - 1]}.{extension}"
        else:
            name = name[:_MAX_FILENAME_LENGTH]
    return name


def _sanitize_content_type(raw: str | None) -> str | None:
    if not raw:
        return None
    content_type = "".join(char for char in raw if char.isprintable()).strip()[:_MAX_CONTENT_TYPE_LENGTH]
    return content_type or None


def _state_for(offset: int, size: int | None) -> UploadState:
    if size is not None and offset == size:
        return UploadState.COMPLETED
    return UploadState.IN_PROGRESS if offset > 0 else UploadState.PENDING


def _truncate(path: Path, length: int) -> None:
    with path.open("r+b") as file:
        file.truncate(length)


class _Hasher(Protocol):
    def update(self, data: bytes, /) -> None: ...

    def digest(self) -> bytes: ...


def _open_at(path: Path, offset: int) -> BufferedRandom:
    file = path.open("r+b")
    file.seek(offset)
    return file


def _write_block(file: BufferedRandom, data: bytes, hasher: _Hasher | None) -> None:
    file.write(data)
    if hasher is not None:
        hasher.update(data)


def _flush_to_disk(file: BufferedRandom) -> None:
    file.flush()
    os.fsync(file.fileno())


class UploadService:
    """
    Stores resumable uploads following the TUS 1.0.0 protocol.

    The database is the source of truth for the acknowledged offset of each upload; the bytes are stored in a flat,
    opaque layout ('<uploads_dir>/<upload_id>.part'). Concurrent operations on the same upload are rejected through
    an in-process lock per upload, which is sufficient because the backend runs as a single process.
    """

    def __init__(
        self,
        uploads_dir: Path,
        max_size: int,
        ttl: timedelta,
        session_factory: Callable[[], AbstractContextManager[Session]] = get_db_session,
    ) -> None:
        self._uploads_dir = uploads_dir
        self._max_size = max_size
        self._ttl = ttl
        self._session_factory = session_factory
        self._locks: dict[UUID, anyio.Lock] = {}
        self._uploads_dir.mkdir(parents=True, exist_ok=True)

    @property
    def max_size(self) -> int:
        """Maximum size in bytes of an upload."""
        return self._max_size

    def path_for(self, upload_id: UUID) -> Path:
        """Path of the file holding the bytes of an upload."""
        return self._uploads_dir / f"{upload_id}.part"

    async def create(
        self,
        size: int | None,
        metadata_header: str | None,
        body: AsyncIterator[bytes] | None = None,
        checksum_header: str | None = None,
    ) -> Upload:
        """
        Create a new upload, optionally storing the first chunk sent along with the creation request.

        Args:
            size: Total size of the upload in bytes, or None if the client deferred it (Upload-Defer-Length).
            metadata_header: Raw 'Upload-Metadata' header.
            body: First chunk of data (creation-with-upload extension), if any.
            checksum_header: Raw 'Upload-Checksum' header covering the first chunk, if any.

        Returns:
            The created upload.

        Raises:
            UploadInvalidError: If the size or the metadata are invalid.
            UploadTooLargeError: If the size exceeds the maximum permitted size.
        """
        if size is not None and size < 0:
            raise UploadInvalidError("Upload-Length must be a non-negative integer.")
        if size is not None and size > self._max_size:
            raise UploadTooLargeError(f"Upload-Length exceeds the maximum size of {self._max_size} bytes.")
        metadata = parse_upload_metadata(metadata_header)
        if checksum_header is not None:
            parse_checksum_header(checksum_header)

        now = datetime.now(UTC)
        upload_id = uuid4()
        row = UploadDB(
            id=str(upload_id),
            filename=sanitize_filename(metadata.get("filename") or metadata.get("name")),
            content_type=_sanitize_content_type(metadata.get("filetype") or metadata.get("type")),
            size=size,
            offset=0,
            state=_state_for(0, size),
            created_at=now,
            updated_at=now,
            expires_at=now + self._ttl,
        )
        # Insert the row before creating the file, so that a file on disk without a row is always an orphan.
        upload = await to_thread.run_sync(self._insert, row)
        await to_thread.run_sync(self.path_for(upload_id).touch)
        logger.info("Created upload {} ('{}', {} bytes)", upload.id, upload.filename, upload.size)

        if body is None:
            return upload
        try:
            return await self.append(upload_id, 0, body, checksum_header=checksum_header)
        except Exception:
            # The client never learns the location of an upload whose creation failed; do not leave it behind.
            await self.delete(upload_id)
            raise

    async def get(self, upload_id: UUID) -> Upload:
        """
        Get an upload in any state.

        Raises:
            UploadNotFoundError: If the upload does not exist.
        """
        return await to_thread.run_sync(self._get, upload_id)

    async def list_all(self) -> list[Upload]:
        """List all the uploads, most recent first."""
        return await to_thread.run_sync(self._list_all)

    async def get_status(self, upload_id: UUID) -> Upload:
        """
        Get the state of an active upload to let a client resume it (TUS 'HEAD').

        When no transfer is in progress, the file on disk is reconciled with the acknowledged offset first.

        Raises:
            UploadNotFoundError: If the upload does not exist.
            UploadGoneError: If the upload has expired or has already been consumed.
        """
        try:
            async with self._exclusive(upload_id):
                return await to_thread.run_sync(self._reconcile, upload_id)
        except UploadLockedError:
            # A PATCH is writing to the file: report the last acknowledged offset without touching the file.
            upload = await self.get(upload_id)
            self._ensure_active(upload)
            return upload

    async def append(
        self,
        upload_id: UUID,
        offset: int,
        chunks: AsyncIterator[bytes],
        checksum_header: str | None = None,
        length: int | None = None,
    ) -> Upload:
        """
        Append bytes to an upload at the given offset (TUS 'PATCH').

        Bytes received before the stream ends are kept, so that a client can resume after a network failure.
        If a checksum is declared, the whole request body is instead discarded when it does not match.

        Args:
            upload_id: ID of the upload.
            offset: Offset declared by the client; must match the acknowledged offset.
            chunks: Request body.
            checksum_header: Raw 'Upload-Checksum' header covering the request body, if any.
            length: Total size of the upload, if the client declares it now (creation-defer-length extension).

        Returns:
            The upload with its new acknowledged offset.

        Raises:
            UploadNotFoundError: If the upload does not exist.
            UploadGoneError: If the upload has expired or has already been consumed.
            UploadLockedError: If another request is operating on the upload.
            UploadOffsetMismatchError: If the offset does not match the acknowledged offset.
            UploadTooLargeError: If the body goes past the declared length or the maximum size.
            UploadInvalidError: If the headers are invalid.
            UploadChecksumMismatchError: If the body does not match the declared checksum.
        """
        checksum = parse_checksum_header(checksum_header) if checksum_header is not None else None
        async with self._exclusive(upload_id):
            upload = await to_thread.run_sync(self._reconcile, upload_id)
            if offset != upload.offset:
                raise UploadOffsetMismatchError(expected=upload.offset, received=offset)

            size = upload.size
            if length is not None:
                if size is not None and length != size:
                    raise UploadInvalidError("Upload-Length cannot be changed once set.")
                if length > self._max_size:
                    raise UploadTooLargeError(f"Upload-Length exceeds the maximum size of {self._max_size} bytes.")
                if length < offset:
                    raise UploadInvalidError("Upload-Length cannot be smaller than the current offset.")
                size = length

            path = self.path_for(upload_id)
            limit = size if size is not None else self._max_size
            try:
                received, digest = await self._write(path, offset, limit, chunks, checksum[0] if checksum else None)
            except (UploadTooLargeError, UploadChecksumMismatchError):
                await to_thread.run_sync(_truncate, path, offset)
                raise
            if checksum is not None and digest != checksum[1]:
                await to_thread.run_sync(_truncate, path, offset)
                raise UploadChecksumMismatchError("The checksum of the received data does not match.")

            return await to_thread.run_sync(
                self._commit_offset, upload_id, offset + received, size, checksum_header if checksum else None
            )

    async def delete(self, upload_id: UUID) -> None:
        """
        Delete an upload and discard its bytes (TUS 'DELETE').

        Raises:
            UploadNotFoundError: If the upload does not exist.
            UploadLockedError: If another request is operating on the upload.
        """
        async with self._exclusive(upload_id):
            await to_thread.run_sync(self._delete, upload_id)
        logger.info("Deleted upload {}", upload_id)

    @asynccontextmanager
    async def consume(self, upload_id: UUID) -> AsyncIterator[ClaimedUpload]:
        """
        Reserve a completed upload for exactly one consumer.

        The consumer is expected to move the file away from the yielded path (e.g. into the project dataset). If the
        consumer raises, the upload is released and can be consumed again; otherwise, it stays 'consumed' and any
        leftover bytes are discarded.

        Raises:
            UploadNotFoundError: If the upload does not exist.
            UploadGoneError: If the upload has expired or has already been consumed.
            UploadLockedError: If another request is operating on the upload.
            UploadNotReadyError: If the upload is not complete.
        """
        async with self._exclusive(upload_id):
            upload = await to_thread.run_sync(self._claim, upload_id)
            path = self.path_for(upload_id)
            try:
                yield ClaimedUpload(upload=upload, path=path)
            except BaseException:
                with anyio.CancelScope(shield=True):
                    # Release regardless of whether the consumer already moved the file away: if it failed after
                    # moving it, the upload must still be releasable (retryable or deletable) instead of staying
                    # stuck in CONSUMED forever.
                    await to_thread.run_sync(self._release, upload_id)
                raise
            await to_thread.run_sync(partial(path.unlink, missing_ok=True))
        logger.info("Consumed upload {} ('{}')", upload.id, upload.filename)

    async def collect_garbage(self) -> int:
        """
        Delete the uploads that have expired or have been consumed, together with their files.

        Uploads with an operation in progress are skipped. Files without a matching upload are also removed.

        Returns:
            The number of uploads deleted.
        """
        now = datetime.now(UTC)
        deleted = 0
        for upload_id in await to_thread.run_sync(self._list_collectable, now):
            try:
                # Holding the lock guarantees that no PATCH or consumption is using the upload meanwhile
                async with self._exclusive(upload_id):
                    deleted += await to_thread.run_sync(self._delete_if_collectable, upload_id, now)
            except UploadLockedError:
                continue
        await to_thread.run_sync(self._remove_orphan_files, now)
        if deleted:
            logger.info("Garbage-collected {} expired or consumed uploads", deleted)
        return deleted

    @asynccontextmanager
    async def _exclusive(self, upload_id: UUID) -> AsyncIterator[None]:
        lock = self._locks.setdefault(upload_id, anyio.Lock())
        # No await between the check and the acquisition: this is atomic within the event loop
        if lock.locked():
            raise UploadLockedError("Another request is currently operating on this upload.")
        lock.acquire_nowait()
        try:
            yield
        finally:
            lock.release()
            # Nobody ever waits for these locks (acquire_nowait), so the entry can be dropped once released.
            self._locks.pop(upload_id, None)

    async def _write(
        self, path: Path, offset: int, limit: int, chunks: AsyncIterator[bytes], algorithm: str | None
    ) -> tuple[int, bytes | None]:
        hasher: _Hasher | None = hashlib.new(algorithm, usedforsecurity=False) if algorithm is not None else None
        received = 0
        buffer = bytearray()
        file = await to_thread.run_sync(_open_at, path, offset)
        try:
            async for chunk in chunks:
                if not chunk:
                    continue
                if offset + received + len(buffer) + len(chunk) > limit:
                    raise UploadTooLargeError("The request body exceeds the upload length or the maximum size.")
                buffer += chunk
                if len(buffer) >= _WRITE_BUFFER_SIZE:
                    await to_thread.run_sync(_write_block, file, bytes(buffer), hasher)
                    received += len(buffer)
                    buffer.clear()
            if buffer:
                await to_thread.run_sync(_write_block, file, bytes(buffer), hasher)
                received += len(buffer)
            await to_thread.run_sync(_flush_to_disk, file)
        finally:
            # Any byte written but not acknowledged in the DB is truncated by the next reconciliation.
            file.close()
        return received, hasher.digest() if hasher is not None else None

    @contextmanager
    def _repository(self) -> Iterator[UploadRepository]:
        with self._session_factory() as session:
            yield UploadRepository(session)

    def _insert(self, row: UploadDB) -> Upload:
        with self._repository() as repo:
            repo.save(row)
            return Upload.model_validate(row)

    def _get(self, upload_id: UUID) -> Upload:
        with self._repository() as repo:
            row = repo.get_by_id(str(upload_id))
            if row is None:
                raise UploadNotFoundError(f"Upload with ID {upload_id} not found.")
            return Upload.model_validate(row)

    def _list_all(self) -> list[Upload]:
        with self._repository() as repo:
            return [Upload.model_validate(row) for row in repo.list_all()]

    @staticmethod
    def _ensure_active(upload: Upload) -> None:
        if upload.state == UploadState.CONSUMED:
            raise UploadGoneError(f"Upload with ID {upload.id} has already been consumed.")
        if upload.expires_at < datetime.now(UTC):
            raise UploadGoneError(f"Upload with ID {upload.id} has expired.")

    def _reconcile(self, upload_id: UUID) -> Upload:
        """Align the file on disk with the acknowledged offset in the DB, which is the source of truth."""
        with self._repository() as repo:
            row = repo.get_by_id(str(upload_id))
            if row is None:
                raise UploadNotFoundError(f"Upload with ID {upload_id} not found.")
            self._ensure_active(Upload.model_validate(row))
            path = self.path_for(upload_id)
            try:
                size_on_disk = path.stat().st_size
            except FileNotFoundError:
                path.touch()
                size_on_disk = 0
            if size_on_disk > row.offset:
                # The process died between writing bytes and acknowledging them: drop the unacknowledged bytes.
                _truncate(path, row.offset)
            elif size_on_disk < row.offset:
                logger.warning(
                    "Upload {} has {} bytes on disk but {} acknowledged; rewinding to the bytes on disk",
                    upload_id,
                    size_on_disk,
                    row.offset,
                )
                row.offset = size_on_disk
                row.state = _state_for(size_on_disk, row.size)
                repo.save(row)
            return Upload.model_validate(row)

    def _commit_offset(self, upload_id: UUID, offset: int, size: int | None, checksum: str | None) -> Upload:
        with self._repository() as repo:
            row = repo.get_by_id(str(upload_id))
            if row is None:
                raise UploadNotFoundError(f"Upload with ID {upload_id} not found.")
            row.offset = offset
            row.size = size
            row.state = _state_for(offset, size)
            if checksum is not None:
                row.checksum = checksum
            row.expires_at = datetime.now(UTC) + self._ttl
            repo.save(row)
            return Upload.model_validate(row)

    def _delete(self, upload_id: UUID) -> None:
        with self._repository() as repo:
            if not repo.delete(str(upload_id)):
                raise UploadNotFoundError(f"Upload with ID {upload_id} not found.")
        self.path_for(upload_id).unlink(missing_ok=True)

    def _claim(self, upload_id: UUID) -> Upload:
        upload = self._reconcile(upload_id)
        if not upload.is_complete:
            raise UploadNotReadyError(f"Upload with ID {upload_id} is not complete ({upload.offset}/{upload.size}).")
        with self._repository() as repo:
            if not repo.claim(str(upload_id), now=datetime.now(UTC)):
                raise UploadGoneError(f"Upload with ID {upload_id} is no longer available.")
        return upload.model_copy(update={"state": UploadState.CONSUMED})

    def _release(self, upload_id: UUID) -> None:
        with self._repository() as repo:
            repo.transition_state(str(upload_id), from_state=UploadState.CONSUMED, to_state=UploadState.COMPLETED)

    def _list_collectable(self, now: datetime) -> list[UUID]:
        with self._repository() as repo:
            return [UUID(row.id) for row in repo.list_collectable(now)]

    def _delete_if_collectable(self, upload_id: UUID, now: datetime) -> bool:
        with self._repository() as repo:
            deleted = repo.delete_if_collectable(str(upload_id), now)
        if deleted:
            self.path_for(upload_id).unlink(missing_ok=True)
        return deleted

    def _remove_orphan_files(self, now: datetime) -> None:
        """Remove the files that do not belong to any upload (e.g. left behind by a crash)."""
        with self._repository() as repo:
            known_ids = {row.id for row in repo.list_all()}
        orphan_cutoff = (now - _ORPHAN_FILE_GRACE_PERIOD).timestamp()
        for path in self._uploads_dir.glob("*.part"):
            if path.stem in known_ids:
                continue
            try:
                if path.stat().st_mtime < orphan_cutoff:
                    path.unlink(missing_ok=True)
            except OSError:
                continue
