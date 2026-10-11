# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from io import BytesIO
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from app.services.source_media_service import SourceMediaService


@pytest.fixture
def fxt_source_media_service(tmp_path: Path) -> SourceMediaService:
    return SourceMediaService(source_media_dir=tmp_path)


class TestSourceMediaService:
    @pytest.mark.asyncio
    async def test_upload_from_path_moves_file(self, tmp_path: Path) -> None:
        service = SourceMediaService(source_media_dir=tmp_path / "source_media")
        source_path = tmp_path / "uploads" / "upload.part"
        source_path.parent.mkdir()
        source_path.write_bytes(b"video-bytes")

        video_path = await service.upload_from_path(filename="../../sample.mp4", source_path=source_path)

        assert not source_path.exists()
        assert video_path.name == "sample.mp4"
        assert video_path.parent.parent == (tmp_path / "source_media").resolve()
        UUID(video_path.parent.name)
        assert video_path.read_bytes() == b"video-bytes"

    @pytest.mark.asyncio
    async def test_upload_from_path_failure_leaves_nothing_behind(self, tmp_path: Path) -> None:
        service = SourceMediaService(source_media_dir=tmp_path / "source_media")
        (tmp_path / "source_media").mkdir()

        with pytest.raises(FileNotFoundError):
            await service.upload_from_path(filename="sample.mp4", source_path=tmp_path / "missing.part")

        assert not any((tmp_path / "source_media").iterdir())

    @pytest.mark.asyncio
    async def test_upload_from_path_rejects_filenames_without_a_usable_name(self, tmp_path: Path) -> None:
        service = SourceMediaService(source_media_dir=tmp_path)

        with pytest.raises(ValueError, match="Invalid filename"):
            await service.upload_from_path(filename="..", source_path=tmp_path / "upload.part")

    @pytest.mark.asyncio
    async def test_upload_writes_file_and_returns_resolved_path(
        self, tmp_path: Path, fxt_source_media_service: SourceMediaService
    ):
        filename = "sample.mp4"
        content = b"fake-video-bytes"

        video_path = await fxt_source_media_service.upload(filename=filename, file_obj=BytesIO(content))

        assert video_path.name == filename
        assert video_path.is_file()
        assert video_path.read_bytes() == content
        assert video_path.is_absolute()
        assert video_path.parent.parent == tmp_path

    @pytest.mark.asyncio
    async def test_upload_generates_unique_paths_for_same_filename(self, fxt_source_media_service: SourceMediaService):
        first_path = await fxt_source_media_service.upload(filename="sample.mp4", file_obj=BytesIO(b"first"))
        second_path = await fxt_source_media_service.upload(filename="sample.mp4", file_obj=BytesIO(b"second"))

        assert first_path != second_path
        assert first_path.read_bytes() == b"first"
        assert second_path.read_bytes() == b"second"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "malicious_filename",
        [
            "../../etc/passwd.mp4",
            "../../../etc/cron.d/evil.mp4",
            "/etc/passwd.mp4",
            "a/b/../../../etc/passwd.mp4",
        ],
    )
    async def test_upload_sanitizes_path_traversal_filenames(
        self, tmp_path: Path, fxt_source_media_service: SourceMediaService, malicious_filename: str
    ):
        video_path = await fxt_source_media_service.upload(filename=malicious_filename, file_obj=BytesIO(b"payload"))

        # The stored file must stay confined to source_media_dir; only the final path
        # component of the supplied filename is honored.
        assert video_path.is_relative_to(tmp_path.resolve())
        assert ".." not in video_path.parts
        assert video_path.name == Path(malicious_filename).name

    @pytest.mark.asyncio
    @pytest.mark.parametrize("empty_filename", ["", ".", "..", "../", "a/../"])
    async def test_upload_rejects_filenames_without_a_usable_name(
        self, fxt_source_media_service: SourceMediaService, empty_filename: str
    ):
        with pytest.raises(ValueError, match="Invalid filename"):
            await fxt_source_media_service.upload(filename=empty_filename, file_obj=BytesIO(b"payload"))

    @pytest.mark.asyncio
    async def test_upload_cleans_up_temp_file_on_failure(
        self, tmp_path: Path, fxt_source_media_service: SourceMediaService, monkeypatch: pytest.MonkeyPatch
    ):
        def _boom(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr("shutil.copyfileobj", _boom)

        with pytest.raises(OSError, match="disk full"):
            await fxt_source_media_service.upload(filename="sample.mp4", file_obj=BytesIO(b"data"))

        # The whole per-upload UUID subdirectory should be removed, not just the temp file.
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.asyncio
    async def test_delete_video_removes_upload_subdirectory(
        self, tmp_path: Path, fxt_source_media_service: SourceMediaService
    ):
        video_path = await fxt_source_media_service.upload(filename="sample.mp4", file_obj=BytesIO(b"data"))
        upload_dir = video_path.parent
        assert upload_dir.is_dir()

        fxt_source_media_service.delete_video(str(video_path))

        assert not upload_dir.exists()
        assert list(tmp_path.iterdir()) == []

    def test_delete_video_noop_when_outside_source_media_dir(
        self, tmp_path: Path, fxt_source_media_service: SourceMediaService
    ):
        outside_dir = tmp_path.parent / f"outside-{tmp_path.name}"
        outside_dir.mkdir()
        outside_file = outside_dir / "video.mp4"
        outside_file.write_bytes(b"data")

        fxt_source_media_service.delete_video(str(outside_file))

        assert outside_file.exists()

    def test_delete_video_noop_when_path_is_directly_under_root(
        self, tmp_path: Path, fxt_source_media_service: SourceMediaService
    ):
        root_level_file = tmp_path / "video.mp4"
        root_level_file.write_bytes(b"data")

        fxt_source_media_service.delete_video(str(root_level_file))

        # No UUID subdirectory boundary, so nothing should be removed (avoids ever
        # rmtree-ing the whole source_media_dir root).
        assert root_level_file.exists()

    def test_delete_video_noop_when_path_is_source_media_root_itself(self, tmp_path: Path):
        # Use a nested dir so a buggy implementation can't accidentally delete pytest's temp root.
        source_media_root = tmp_path / "source_media"
        source_media_root.mkdir()

        service = SourceMediaService(source_media_dir=source_media_root)
        service.delete_video(str(source_media_root))

        assert source_media_root.exists()

    @pytest.mark.asyncio
    async def test_delete_video_propagates_oserror(
        self,
        fxt_source_media_service: SourceMediaService,
        monkeypatch: pytest.MonkeyPatch,
    ):
        video_path = await fxt_source_media_service.upload(filename="sample.mp4", file_obj=BytesIO(b"data"))

        def _boom(*args, **kwargs):
            raise OSError("permission denied")

        monkeypatch.setattr("shutil.rmtree", _boom)

        with pytest.raises(OSError, match="permission denied"):
            fxt_source_media_service.delete_video(str(video_path))

    @pytest.mark.asyncio
    async def test_find_upload_by_id_returns_stored_file(self, fxt_source_media_service: SourceMediaService):
        video_path = await fxt_source_media_service.upload(filename="sample.mp4", file_obj=BytesIO(b"data"))
        source_media_id = UUID(video_path.parent.name)

        found = fxt_source_media_service.find_upload_by_id(source_media_id)

        assert found == video_path

    def test_find_upload_by_id_returns_none_for_unknown_uuid(self, fxt_source_media_service: SourceMediaService):
        assert fxt_source_media_service.find_upload_by_id(uuid4()) is None

    def test_find_upload_by_id_returns_none_for_empty_or_ambiguous_subdirectory(
        self, tmp_path: Path, fxt_source_media_service: SourceMediaService
    ):
        empty_dir = tmp_path / str(uuid4())
        empty_dir.mkdir()
        ambiguous_dir = tmp_path / str(uuid4())
        ambiguous_dir.mkdir()
        (ambiguous_dir / "first.mp4").write_bytes(b"one")
        (ambiguous_dir / "second.mp4").write_bytes(b"two")

        assert fxt_source_media_service.find_upload_by_id(UUID(empty_dir.name)) is None
        assert fxt_source_media_service.find_upload_by_id(UUID(ambiguous_dir.name)) is None

    @pytest.mark.asyncio
    async def test_quarantine_upload_moves_upload_aside_and_restore_reverts(
        self, tmp_path: Path, fxt_source_media_service: SourceMediaService
    ):
        video_path = await fxt_source_media_service.upload(filename="sample.mp4", file_obj=BytesIO(b"data"))
        upload_dir = video_path.parent

        quarantined_path = fxt_source_media_service.quarantine_upload(video_path)

        assert not upload_dir.exists()
        assert quarantined_path.is_file()
        assert quarantined_path.name == video_path.name
        assert quarantined_path.parent.parent == tmp_path

        fxt_source_media_service.restore_quarantined_upload(quarantined_path)

        assert video_path.is_file()
        assert set(tmp_path.iterdir()) == {upload_dir}

    @pytest.mark.asyncio
    async def test_delete_quarantined_upload_removes_directory(
        self, tmp_path: Path, fxt_source_media_service: SourceMediaService
    ):
        video_path = await fxt_source_media_service.upload(filename="sample.mp4", file_obj=BytesIO(b"data"))

        quarantined_path = fxt_source_media_service.quarantine_upload(video_path)
        fxt_source_media_service.delete_quarantined_upload(quarantined_path)

        assert list(tmp_path.iterdir()) == []

    def test_quarantine_upload_rejects_paths_outside_upload_layout(
        self, tmp_path: Path, fxt_source_media_service: SourceMediaService
    ):
        root_level_file = tmp_path / "video.mp4"
        root_level_file.write_bytes(b"data")
        outside_file = tmp_path.parent / f"outside-{tmp_path.name}" / "video.mp4"
        outside_file.parent.mkdir()
        outside_file.write_bytes(b"data")

        with pytest.raises(FileNotFoundError):
            fxt_source_media_service.quarantine_upload(str(root_level_file))
        with pytest.raises(FileNotFoundError):
            fxt_source_media_service.quarantine_upload(str(outside_file))
        with pytest.raises(FileNotFoundError):
            fxt_source_media_service.quarantine_upload(str(tmp_path / "missing-uuid" / "video.mp4"))

        assert root_level_file.is_file()
        assert outside_file.is_file()
