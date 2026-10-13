# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import base64
import hashlib
import os
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import anyio
import pytest

from app.models import UploadState
from app.services.upload_service import (
    UploadChecksumMismatchError,
    UploadGoneError,
    UploadInvalidError,
    UploadLockedError,
    UploadNotFoundError,
    UploadNotReadyError,
    UploadOffsetMismatchError,
    UploadService,
    UploadTooLargeError,
    parse_checksum_header,
    parse_upload_metadata,
    sanitize_filename,
)


def _b64(value: str | bytes) -> str:
    raw = value.encode() if isinstance(value, str) else value
    return base64.b64encode(raw).decode("ascii")


def _metadata(**values: str) -> str:
    return ",".join(f"{key} {_b64(value)}" for key, value in values.items())


def _checksum(algorithm: str, data: bytes) -> str:
    return f"{algorithm} {_b64(hashlib.new(algorithm, data).digest())}"


async def _chunks(*chunks: bytes) -> AsyncIterator[bytes]:
    for chunk in chunks:
        yield chunk


def _expire(service: UploadService, upload_id) -> None:
    with service._repository() as repo:
        row = repo.get_by_id(str(upload_id))
        assert row is not None
        row.expires_at = datetime.now(UTC) - timedelta(seconds=1)
        repo.save(row)


class TestParseUploadMetadata:
    def test_parse(self) -> None:
        header = f"filename {_b64('grapes.mp4')},filetype {_b64('video/mp4')},is_confidential"

        assert parse_upload_metadata(header) == {
            "filename": "grapes.mp4",
            "filetype": "video/mp4",
            "is_confidential": "",
        }

    @pytest.mark.parametrize("header", [None, "", "   "])
    def test_parse_empty(self, header: str | None) -> None:
        assert parse_upload_metadata(header) == {}

    @pytest.mark.parametrize(
        "header",
        [
            "filename not-base64!",
            "filename a b",
            f"filename {_b64('a')},filename {_b64('b')}",
            f"file\tname {_b64('a')}",
            f"filename {_b64(b'\xff\xfe')}",
            "x" * 10_000,
        ],
    )
    def test_parse_malformed(self, header: str) -> None:
        with pytest.raises(UploadInvalidError):
            parse_upload_metadata(header)


class TestParseChecksumHeader:
    def test_parse(self) -> None:
        assert parse_checksum_header(_checksum("sha1", b"abc")) == ("sha1", hashlib.sha1(b"abc").digest())

    @pytest.mark.parametrize("header", ["sha1", "crc32 AAAA", "sha1 not-base64!", "sha1 a b"])
    def test_parse_invalid(self, header: str) -> None:
        with pytest.raises(UploadInvalidError):
            parse_checksum_header(header)


class TestSanitizeFilename:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("grapes.mp4", "grapes.mp4"),
            ("../../etc/passwd", "passwd"),
            ("C:\\Users\\me\\video.mp4", "video.mp4"),
            (".hidden.jpg", "hidden.jpg"),
            ("bad\x00name\n.png", "badname.png"),
            ("", "upload"),
            (None, "upload"),
            ("..", "upload"),
        ],
    )
    def test_sanitize(self, raw: str | None, expected: str) -> None:
        assert sanitize_filename(raw) == expected

    def test_length_is_capped_and_extension_kept(self) -> None:
        name = sanitize_filename("a" * 1000 + ".jpeg")

        assert len(name) == 255
        assert name.endswith(".jpeg")


class TestUploadService:
    @pytest.mark.asyncio
    async def test_create(self, fxt_upload_service: UploadService) -> None:
        upload = await fxt_upload_service.create(
            size=10, metadata_header=_metadata(filename="../grapes.mp4", filetype="video/mp4")
        )

        assert upload.filename == "grapes.mp4"
        assert upload.content_type == "video/mp4"
        assert upload.size == 10
        assert upload.offset == 0
        assert upload.state == UploadState.PENDING
        assert upload.expires_at > datetime.now(UTC)
        path = fxt_upload_service.path_for(upload.id)
        assert path.name == f"{upload.id}.part"
        assert path.read_bytes() == b""
        assert await fxt_upload_service.get(upload.id) == upload

    @pytest.mark.asyncio
    async def test_create_too_large(self, fxt_upload_service: UploadService) -> None:
        with pytest.raises(UploadTooLargeError):
            await fxt_upload_service.create(size=fxt_upload_service.max_size + 1, metadata_header=None)

        assert await fxt_upload_service.list_all() == []

    @pytest.mark.asyncio
    async def test_create_invalid_metadata(self, fxt_upload_service: UploadService) -> None:
        with pytest.raises(UploadInvalidError):
            await fxt_upload_service.create(size=1, metadata_header="filename %%%")

        assert await fxt_upload_service.list_all() == []

    @pytest.mark.asyncio
    async def test_create_with_upload(self, fxt_upload_service: UploadService) -> None:
        upload = await fxt_upload_service.create(size=6, metadata_header=None, body=_chunks(b"abc"))

        assert upload.offset == 3
        assert upload.state == UploadState.IN_PROGRESS
        assert fxt_upload_service.path_for(upload.id).read_bytes() == b"abc"

    @pytest.mark.asyncio
    async def test_create_with_upload_failure_leaves_nothing_behind(self, fxt_upload_service: UploadService) -> None:
        with pytest.raises(UploadTooLargeError):
            await fxt_upload_service.create(size=2, metadata_header=None, body=_chunks(b"abc"))

        assert await fxt_upload_service.list_all() == []
        assert list(fxt_upload_service.path_for(uuid4()).parent.iterdir()) == []

    @pytest.mark.asyncio
    async def test_append_and_resume(self, fxt_upload_service: UploadService) -> None:
        upload = await fxt_upload_service.create(size=6, metadata_header=None)

        upload = await fxt_upload_service.append(upload.id, 0, _chunks(b"ab", b"c"))
        assert upload.offset == 3
        assert upload.state == UploadState.IN_PROGRESS

        # The client resumes from the offset reported by HEAD
        status = await fxt_upload_service.get_status(upload.id)
        assert status.offset == 3
        upload = await fxt_upload_service.append(upload.id, status.offset, _chunks(b"def"))

        assert upload.offset == 6
        assert upload.state == UploadState.COMPLETED
        assert upload.is_complete
        assert fxt_upload_service.path_for(upload.id).read_bytes() == b"abcdef"

    @pytest.mark.asyncio
    async def test_append_extends_expiration(self, fxt_upload_service: UploadService) -> None:
        upload = await fxt_upload_service.create(size=6, metadata_header=None)

        updated = await fxt_upload_service.append(upload.id, 0, _chunks(b"a"))

        assert updated.expires_at > upload.expires_at

    @pytest.mark.asyncio
    async def test_append_offset_mismatch(self, fxt_upload_service: UploadService) -> None:
        upload = await fxt_upload_service.create(size=6, metadata_header=None, body=_chunks(b"abc"))

        with pytest.raises(UploadOffsetMismatchError) as error:
            await fxt_upload_service.append(upload.id, 0, _chunks(b"abc"))

        assert error.value.expected == 3
        assert fxt_upload_service.path_for(upload.id).read_bytes() == b"abc"

    @pytest.mark.asyncio
    async def test_append_past_upload_length_is_rejected(self, fxt_upload_service: UploadService) -> None:
        upload = await fxt_upload_service.create(size=4, metadata_header=None, body=_chunks(b"ab"))

        with pytest.raises(UploadTooLargeError):
            await fxt_upload_service.append(upload.id, 2, _chunks(b"c", b"de"))

        assert (await fxt_upload_service.get(upload.id)).offset == 2
        assert fxt_upload_service.path_for(upload.id).read_bytes() == b"ab"

    @pytest.mark.asyncio
    async def test_append_keeps_bytes_received_before_disconnection(self, fxt_upload_service: UploadService) -> None:
        upload = await fxt_upload_service.create(size=6, metadata_header=None)

        async def interrupted() -> AsyncIterator[bytes]:
            yield b"abcd"

        upload = await fxt_upload_service.append(upload.id, 0, interrupted())

        assert upload.offset == 4
        assert upload.state == UploadState.IN_PROGRESS

    @pytest.mark.asyncio
    async def test_append_with_valid_checksum(self, fxt_upload_service: UploadService) -> None:
        upload = await fxt_upload_service.create(size=3, metadata_header=None)
        checksum = _checksum("sha256", b"abc")

        upload = await fxt_upload_service.append(upload.id, 0, _chunks(b"abc"), checksum_header=checksum)

        assert upload.is_complete
        assert upload.checksum == checksum

    @pytest.mark.asyncio
    async def test_append_with_checksum_mismatch_discards_bytes(self, fxt_upload_service: UploadService) -> None:
        upload = await fxt_upload_service.create(size=6, metadata_header=None, body=_chunks(b"abc"))

        with pytest.raises(UploadChecksumMismatchError):
            await fxt_upload_service.append(upload.id, 3, _chunks(b"def"), checksum_header=_checksum("sha1", b"xyz"))

        assert (await fxt_upload_service.get(upload.id)).offset == 3
        assert fxt_upload_service.path_for(upload.id).read_bytes() == b"abc"

    @pytest.mark.asyncio
    async def test_append_with_deferred_length(self, fxt_upload_service: UploadService) -> None:
        upload = await fxt_upload_service.create(size=None, metadata_header=None)
        upload = await fxt_upload_service.append(upload.id, 0, _chunks(b"abc"))
        assert upload.size is None
        assert upload.state == UploadState.IN_PROGRESS

        with pytest.raises(UploadInvalidError):
            await fxt_upload_service.append(upload.id, 3, _chunks(), length=2)
        with pytest.raises(UploadTooLargeError):
            await fxt_upload_service.append(upload.id, 3, _chunks(), length=fxt_upload_service.max_size + 1)
        upload = await fxt_upload_service.append(upload.id, 3, _chunks(b"de"), length=5)

        assert upload.size == 5
        assert upload.state == UploadState.COMPLETED

    @pytest.mark.asyncio
    async def test_append_cannot_change_length(self, fxt_upload_service: UploadService) -> None:
        upload = await fxt_upload_service.create(size=5, metadata_header=None)

        with pytest.raises(UploadInvalidError):
            await fxt_upload_service.append(upload.id, 0, _chunks(b"a"), length=6)

    @pytest.mark.asyncio
    async def test_parallel_append_is_locked(self, fxt_upload_service: UploadService) -> None:
        upload = await fxt_upload_service.create(size=6, metadata_header=None)
        first_chunk_written = anyio.Event()
        release = anyio.Event()

        async def slow() -> AsyncIterator[bytes]:
            yield b"abc"
            first_chunk_written.set()
            await release.wait()

        async with anyio.create_task_group() as tg:
            tg.start_soon(fxt_upload_service.append, upload.id, 0, slow())  # pyrefly: ignore[bad-argument-type]
            await first_chunk_written.wait()

            with pytest.raises(UploadLockedError):
                await fxt_upload_service.append(upload.id, 0, _chunks(b"abc"))
            with pytest.raises(UploadLockedError):
                await fxt_upload_service.delete(upload.id)
            # HEAD during a transfer reports the last acknowledged offset, without touching the file
            assert (await fxt_upload_service.get_status(upload.id)).offset == 0
            release.set()

        assert (await fxt_upload_service.get(upload.id)).offset == 3

    @pytest.mark.asyncio
    async def test_reconcile_truncates_unacknowledged_bytes(self, fxt_upload_service: UploadService) -> None:
        upload = await fxt_upload_service.create(size=6, metadata_header=None, body=_chunks(b"abc"))
        # Simulate a crash between writing bytes and acknowledging them in the DB
        with fxt_upload_service.path_for(upload.id).open("ab") as file:
            file.write(b"de")

        status = await fxt_upload_service.get_status(upload.id)

        assert status.offset == 3
        assert fxt_upload_service.path_for(upload.id).read_bytes() == b"abc"

    @pytest.mark.asyncio
    async def test_reconcile_rewinds_to_bytes_on_disk(self, fxt_upload_service: UploadService) -> None:
        upload = await fxt_upload_service.create(size=6, metadata_header=None, body=_chunks(b"abc"))
        with fxt_upload_service.path_for(upload.id).open("r+b") as file:
            file.truncate(1)

        status = await fxt_upload_service.get_status(upload.id)

        assert status.offset == 1

    @pytest.mark.asyncio
    async def test_expired_upload_is_gone(self, fxt_upload_service: UploadService) -> None:
        upload = await fxt_upload_service.create(size=3, metadata_header=None)
        _expire(fxt_upload_service, upload.id)

        with pytest.raises(UploadGoneError):
            await fxt_upload_service.get_status(upload.id)
        with pytest.raises(UploadGoneError):
            await fxt_upload_service.append(upload.id, 0, _chunks(b"abc"))

    @pytest.mark.asyncio
    async def test_not_found(self, fxt_upload_service: UploadService) -> None:
        with pytest.raises(UploadNotFoundError):
            await fxt_upload_service.get(uuid4())
        with pytest.raises(UploadNotFoundError):
            await fxt_upload_service.get_status(uuid4())
        with pytest.raises(UploadNotFoundError):
            await fxt_upload_service.append(uuid4(), 0, _chunks(b"a"))
        with pytest.raises(UploadNotFoundError):
            await fxt_upload_service.delete(uuid4())

    @pytest.mark.asyncio
    async def test_delete(self, fxt_upload_service: UploadService) -> None:
        upload = await fxt_upload_service.create(size=6, metadata_header=None, body=_chunks(b"abc"))

        await fxt_upload_service.delete(upload.id)

        assert not fxt_upload_service.path_for(upload.id).exists()
        with pytest.raises(UploadNotFoundError):
            await fxt_upload_service.get(upload.id)

    @pytest.mark.asyncio
    async def test_list_all(self, fxt_upload_service: UploadService) -> None:
        first = await fxt_upload_service.create(size=1, metadata_header=None)
        second = await fxt_upload_service.create(size=1, metadata_header=None)

        assert [upload.id for upload in await fxt_upload_service.list_all()] == [second.id, first.id]

    @pytest.mark.asyncio
    async def test_consume(self, fxt_upload_service: UploadService, tmp_path) -> None:
        upload = await fxt_upload_service.create(size=3, metadata_header=None, body=_chunks(b"abc"))
        target = tmp_path / "target.bin"

        async with fxt_upload_service.consume(upload.id) as claimed:
            assert claimed.upload.state == UploadState.CONSUMED
            assert claimed.path == fxt_upload_service.path_for(upload.id)
            claimed.path.rename(target)

        assert target.read_bytes() == b"abc"
        assert (await fxt_upload_service.get(upload.id)).state == UploadState.CONSUMED
        # An upload can be consumed only once
        with pytest.raises(UploadGoneError):
            async with fxt_upload_service.consume(upload.id):
                pytest.fail("Consumed twice")
        with pytest.raises(UploadGoneError):
            await fxt_upload_service.get_status(upload.id)

    @pytest.mark.asyncio
    async def test_consume_discards_leftover_file(self, fxt_upload_service: UploadService) -> None:
        upload = await fxt_upload_service.create(size=3, metadata_header=None, body=_chunks(b"abc"))

        async with fxt_upload_service.consume(upload.id):
            pass

        assert not fxt_upload_service.path_for(upload.id).exists()

    @pytest.mark.asyncio
    async def test_consume_failure_releases_upload(self, fxt_upload_service: UploadService) -> None:
        upload = await fxt_upload_service.create(size=3, metadata_header=None, body=_chunks(b"abc"))

        with pytest.raises(RuntimeError):
            async with fxt_upload_service.consume(upload.id):
                raise RuntimeError("consumer failed")

        assert (await fxt_upload_service.get(upload.id)).state == UploadState.COMPLETED
        async with fxt_upload_service.consume(upload.id) as claimed:
            assert claimed.path.read_bytes() == b"abc"

    @pytest.mark.asyncio
    async def test_consume_failure_after_file_moved_still_releases_upload(
        self, fxt_upload_service: UploadService
    ) -> None:
        """A consumer that moves the file away before failing must not leave the upload stuck as CONSUMED."""
        upload = await fxt_upload_service.create(size=3, metadata_header=None, body=_chunks(b"abc"))

        with pytest.raises(RuntimeError):
            async with fxt_upload_service.consume(upload.id) as claimed:
                claimed.path.unlink()  # simulate the consumer moving the file away
                raise RuntimeError("consumer failed after moving the file")

        assert (await fxt_upload_service.get(upload.id)).state == UploadState.COMPLETED

    @pytest.mark.asyncio
    async def test_consume_incomplete_upload(self, fxt_upload_service: UploadService) -> None:
        upload = await fxt_upload_service.create(size=6, metadata_header=None, body=_chunks(b"abc"))

        with pytest.raises(UploadNotReadyError):
            async with fxt_upload_service.consume(upload.id):
                pytest.fail("Consumed an incomplete upload")

    @pytest.mark.asyncio
    async def test_consume_expired_upload(self, fxt_upload_service: UploadService) -> None:
        upload = await fxt_upload_service.create(size=3, metadata_header=None, body=_chunks(b"abc"))
        _expire(fxt_upload_service, upload.id)

        with pytest.raises(UploadGoneError):
            async with fxt_upload_service.consume(upload.id):
                pytest.fail("Consumed an expired upload")

    @pytest.mark.asyncio
    async def test_concurrent_consume_is_locked(self, fxt_upload_service: UploadService) -> None:
        upload = await fxt_upload_service.create(size=3, metadata_header=None, body=_chunks(b"abc"))

        async with fxt_upload_service.consume(upload.id):
            with pytest.raises(UploadLockedError):
                async with fxt_upload_service.consume(upload.id):
                    pytest.fail("Consumed twice")

    @pytest.mark.asyncio
    async def test_collect_garbage(self, fxt_upload_service: UploadService) -> None:
        active = await fxt_upload_service.create(size=3, metadata_header=None, body=_chunks(b"a"))
        expired = await fxt_upload_service.create(size=3, metadata_header=None, body=_chunks(b"a"))
        _expire(fxt_upload_service, expired.id)
        consumed = await fxt_upload_service.create(size=1, metadata_header=None, body=_chunks(b"a"))
        async with fxt_upload_service.consume(consumed.id):
            pass
        uploads_dir = fxt_upload_service.path_for(active.id).parent
        old_orphan = uploads_dir / f"{uuid4()}.part"
        old_orphan.write_bytes(b"x")
        old_time = time.time() - 2 * 3600
        os.utime(old_orphan, (old_time, old_time))
        recent_orphan = uploads_dir / f"{uuid4()}.part"
        recent_orphan.write_bytes(b"x")

        deleted = await fxt_upload_service.collect_garbage()

        assert deleted == 2
        assert [upload.id for upload in await fxt_upload_service.list_all()] == [active.id]
        assert fxt_upload_service.path_for(active.id).exists()
        assert not fxt_upload_service.path_for(expired.id).exists()
        assert not old_orphan.exists()
        assert recent_orphan.exists()

    @pytest.mark.asyncio
    async def test_collect_garbage_skips_uploads_in_use(self, fxt_upload_service: UploadService) -> None:
        upload = await fxt_upload_service.create(size=3, metadata_header=None, body=_chunks(b"abc"))

        async with fxt_upload_service.consume(upload.id):
            # The upload is 'consumed' but its consumer is still running
            assert await fxt_upload_service.collect_garbage() == 0

        assert await fxt_upload_service.collect_garbage() == 1
