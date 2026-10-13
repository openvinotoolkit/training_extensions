# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from sqlalchemy.orm import Session

from app.models import SourceType
from app.models.source import SourceAdapter, VideoFileConfig
from app.services.base import ResourceInUseError, ResourceNotFoundError
from app.services.source_media_service import SourceMediaService
from app.services.source_service import SourceService


@pytest.fixture
def fxt_source_media_service(tmp_path: Path) -> SourceMediaService:
    return SourceMediaService(source_media_dir=tmp_path)


@pytest.fixture
def fxt_source_service(fxt_source_media_service: SourceMediaService) -> SourceService:
    return SourceService(db_session=MagicMock(spec=Session), source_media_service=fxt_source_media_service)


def _store_upload(source_media_dir: Path, filename: str) -> Path:
    upload_dir = source_media_dir / str(uuid4())
    upload_dir.mkdir(parents=True)
    upload_path = upload_dir / filename
    upload_path.write_bytes(b"fake-video-bytes")
    return upload_path.resolve()


def _source_media_id(video_path: Path) -> UUID:
    return UUID(video_path.parent.name)


def _video_file_source(video_path: str):
    return SourceAdapter.validate_python(
        {
            "id": str(uuid4()),
            "name": "video source",
            "source_type": SourceType.VIDEO_FILE,
            "config_data": VideoFileConfig(video_path=video_path),
        }
    )


class TestDeleteUnreferencedMedia:
    def test_deletes_unreferenced_upload(
        self,
        fxt_source_media_service: SourceMediaService,
        fxt_source_service: SourceService,
        monkeypatch: pytest.MonkeyPatch,
    ):
        source_media_dir = fxt_source_media_service._source_media_dir
        upload_path = _store_upload(source_media_dir, "sample.mp4")
        other_path = _store_upload(source_media_dir, "other.mp4")
        referenced_path = _store_upload(source_media_dir, "referenced.mp4")
        monkeypatch.setattr(fxt_source_service, "list_all", lambda: [_video_file_source(str(referenced_path))])

        deleted_path = fxt_source_service.delete_unreferenced_media(_source_media_id(upload_path))

        assert deleted_path == str(upload_path)
        assert not upload_path.parent.exists()
        assert other_path.is_file()
        assert referenced_path.is_file()

    def test_raises_in_use_and_keeps_file_when_referenced(
        self,
        fxt_source_media_service: SourceMediaService,
        fxt_source_service: SourceService,
        monkeypatch: pytest.MonkeyPatch,
    ):
        source_media_dir = fxt_source_media_service._source_media_dir
        upload_path = _store_upload(source_media_dir, "sample.mp4")
        monkeypatch.setattr(fxt_source_service, "list_all", lambda: [_video_file_source(str(upload_path))])

        with pytest.raises(ResourceInUseError, match="in use by a video_file source"):
            fxt_source_service.delete_unreferenced_media(_source_media_id(upload_path))

        assert upload_path.is_file()

    def test_detects_equivalent_path_spellings(
        self,
        fxt_source_media_service: SourceMediaService,
        fxt_source_service: SourceService,
        monkeypatch: pytest.MonkeyPatch,
    ):
        source_media_dir = fxt_source_media_service._source_media_dir
        upload_path = _store_upload(source_media_dir, "sample.mp4")
        # Same location, but spelled with a '..' segment: a raw-string comparison would
        # miss it, a resolved-path comparison must not.
        equivalent_spelling = f"{upload_path.parent}/../{upload_path.parent.name}/{upload_path.name}"
        monkeypatch.setattr(fxt_source_service, "list_all", lambda: [_video_file_source(equivalent_spelling)])

        with pytest.raises(ResourceInUseError, match="in use by a video_file source"):
            fxt_source_service.delete_unreferenced_media(_source_media_id(upload_path))

        assert upload_path.is_file()

    def test_restores_upload_when_reference_appears_concurrently(
        self,
        fxt_source_media_service: SourceMediaService,
        fxt_source_service: SourceService,
        monkeypatch: pytest.MonkeyPatch,
    ):
        # First reference check passes; a source committing between the check and the
        # deletion must make the endpoint refuse and restore the quarantined upload.
        source_media_dir = fxt_source_media_service._source_media_dir
        upload_path = _store_upload(source_media_dir, "sample.mp4")
        list_all_results = iter([[], [_video_file_source(str(upload_path))]])
        monkeypatch.setattr(fxt_source_service, "list_all", lambda: next(list_all_results))

        with pytest.raises(ResourceInUseError, match="in use by a video_file source"):
            fxt_source_service.delete_unreferenced_media(_source_media_id(upload_path))

        assert upload_path.is_file()
        assert set(source_media_dir.iterdir()) == {upload_path.parent}

    def test_raises_not_found_for_unknown_uuid(self, fxt_source_service: SourceService):
        with pytest.raises(ResourceNotFoundError):
            fxt_source_service.delete_unreferenced_media(uuid4())

    def test_raises_not_found_when_upload_disappears(
        self,
        fxt_source_media_service: SourceMediaService,
        fxt_source_service: SourceService,
        monkeypatch: pytest.MonkeyPatch,
    ):
        vanished_path = fxt_source_media_service._source_media_dir / str(uuid4()) / "sample.mp4"
        monkeypatch.setattr(fxt_source_media_service, "find_upload_by_id", lambda _id: vanished_path)
        monkeypatch.setattr(fxt_source_service, "list_all", list)

        with pytest.raises(ResourceNotFoundError):
            fxt_source_service.delete_unreferenced_media(uuid4())

    def test_raises_not_found_without_media_service(self):
        source_service = SourceService(db_session=MagicMock(spec=Session), source_media_service=None)

        with pytest.raises(ResourceNotFoundError):
            source_service.delete_unreferenced_media(uuid4())
