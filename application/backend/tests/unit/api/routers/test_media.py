# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from unittest.mock import ANY, MagicMock, PropertyMock, call, patch
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest
from fastapi import status
from PIL import Image as PILImage

from app.api.dependencies import (
    get_dataset_service,
    get_dataset_view_service,
    get_inference_media_limit,
    get_media_prediction_service,
    get_media_segment_service,
    get_media_service,
    get_project_service,
)
from app.api.schemas.media import ImageView, MediaViewAdapter, SetMediaAnnotations, VideoFrameView, VideoView
from app.models import (
    BatchInferenceMedia,
    BatchInferencePrediction,
    BatchInferenceResult,
    DatasetItem,
    DatasetItemAnnotation,
    DatasetItemAnnotationStatus,
    DatasetItemSubset,
    Image,
    LabelReference,
    MediaType,
    Rectangle,
    Video,
    VideoFrame,
)
from app.models.media import (
    ImageFormat,
    MediaListPredictionRequest,
    MediaPredictionRequest,
    MediaSortBy,
    SortDirection,
    VideoFormat,
    VideoRange,
)
from app.models.system import DeviceInfo, DeviceType
from app.services import (
    DatasetService,
    DatasetViewService,
    MediaPredictionService,
    MediaService,
    ProjectService,
    ResourceNotFoundError,
    ResourceType,
)
from app.services.dataset_service import AnnotationValidationError, SubsetAlreadyAssignedError
from app.services.inference import InferenceBusyError
from app.services.media_numpy_loader import BinaryNotFoundError
from app.services.media_prediction_service import VideoRangeError
from app.services.media_service import ImageMetadata, MediaFilters
from app.services.sam import MediaSegmentService


@pytest.fixture
def fxt_image_media():
    return Image(
        id=uuid4(),
        type=MediaType.IMAGE,
        project_id=uuid4(),
        name="test_image",
        format=ImageFormat.JPG,
        width=1024,
        height=768,
        size=2048,
        source_id=uuid4(),
    )


@pytest.fixture
def fxt_video_media():
    return Video(
        id=uuid4(),
        type=MediaType.VIDEO,
        project_id=uuid4(),
        name="test_video",
        format=VideoFormat.MP4,
        width=1024,
        height=768,
        size=2048,
        fps=25,
        frame_count=1000,
        annotated_frame_count=100,
        source_id=uuid4(),
    )


@pytest.fixture
def fxt_video_frame_media():
    return VideoFrame(
        id=uuid4(),
        type=MediaType.VIDEO_FRAME,
        project_id=uuid4(),
        name="test_video_frame_3",
        format=VideoFormat.MP4,
        width=1024,
        height=768,
        size=2048,
        frame_index=3,
        video_id=uuid4(),
        source_id=uuid4(),
    )


@pytest.fixture
def fxt_project_service(fxt_app) -> MagicMock:
    project_service = MagicMock(spec=ProjectService)
    fxt_app.dependency_overrides[get_project_service] = lambda: project_service
    return project_service


@pytest.fixture
def fxt_media_service(fxt_app) -> MagicMock:
    media_service = MagicMock(spec=MediaService)
    fxt_app.dependency_overrides[get_media_service] = lambda: media_service
    return media_service


@pytest.fixture
def fxt_dataset_service(fxt_app) -> MagicMock:
    dataset_service = MagicMock(spec=DatasetService)
    fxt_app.dependency_overrides[get_dataset_service] = lambda: dataset_service
    return dataset_service


@pytest.fixture
def fxt_dataset_view_service(fxt_app) -> MagicMock:
    dataset_view_service = MagicMock(spec=DatasetViewService)
    fxt_app.dependency_overrides[get_dataset_view_service] = lambda: dataset_view_service
    return dataset_view_service


@pytest.fixture
def fxt_media_prediction_service(fxt_app) -> MagicMock:
    media_prediction_service = MagicMock(spec=MediaPredictionService)
    fxt_app.dependency_overrides[get_media_prediction_service] = lambda: media_prediction_service
    return media_prediction_service


@pytest.fixture
def fxt_media_segment_service(fxt_app) -> MagicMock:
    media_segment_service = MagicMock(spec=MediaSegmentService)
    fxt_app.dependency_overrides[get_media_segment_service] = lambda: media_segment_service
    return media_segment_service


@pytest.fixture
def fxt_inference_media_limit(fxt_app) -> Callable[[int], None]:
    def set_limit(limit: int) -> None:
        fxt_app.dependency_overrides[get_inference_media_limit] = lambda: limit

    return set_limit


def test_convert_image_to_view(fxt_image_media) -> None:
    view = MediaViewAdapter.validate_python(fxt_image_media, from_attributes=True)
    assert view == ImageView(
        id=fxt_image_media.id,
        name=fxt_image_media.name,
        type=fxt_image_media.type,
        format=fxt_image_media.format,
        width=fxt_image_media.width,
        height=fxt_image_media.height,
        size=fxt_image_media.size,
        source_id=fxt_image_media.source_id,
    )


def test_convert_video_to_view(fxt_video_media) -> None:
    view = MediaViewAdapter.validate_python(fxt_video_media, from_attributes=True)
    assert view == VideoView(
        id=fxt_video_media.id,
        name=fxt_video_media.name,
        type=fxt_video_media.type,
        format=fxt_video_media.format,
        width=fxt_video_media.width,
        height=fxt_video_media.height,
        size=fxt_video_media.size,
        fps=fxt_video_media.fps,
        frame_count=fxt_video_media.frame_count,
        annotated_frame_count=fxt_video_media.annotated_frame_count,
        source_id=fxt_video_media.source_id,
        duration=fxt_video_media.duration,
    )


def test_convert_video_frame_to_view(fxt_video_frame_media) -> None:
    view = MediaViewAdapter.validate_python(fxt_video_frame_media, from_attributes=True)
    assert view == VideoFrameView(
        id=fxt_video_frame_media.id,
        name=fxt_video_frame_media.name,
        type=fxt_video_frame_media.type,
        format=fxt_video_frame_media.format,
        width=fxt_video_frame_media.width,
        height=fxt_video_frame_media.height,
        size=fxt_video_frame_media.size,
        frame_index=fxt_video_frame_media.frame_index,
        video_id=fxt_video_frame_media.video_id,
        source_id=fxt_video_frame_media.source_id,
    )


class TestMediaEndpoints:
    def test_create_media_no_file(
        self, fxt_get_project, fxt_image_media, fxt_media_service, fxt_dataset_service, fxt_client
    ):
        response = fxt_client.post(f"/api/projects/{uuid4()}/dataset/media")

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
        fxt_media_service.create_image.assert_not_called()
        fxt_media_service.create_video.assert_not_called()
        fxt_dataset_service.create_dataset_item.assert_not_called()

    @pytest.mark.parametrize("image_format", ["jpg", "jpeg", "bmp", "png", "tiff", "tif", "webp", "jfif"])
    def test_create_image_success(
        self,
        fxt_get_project,
        fxt_image_media,
        fxt_media_service,
        fxt_dataset_service,
        fxt_client,
        image_format,
    ):
        fxt_media_service.create_image.return_value = fxt_image_media

        response = fxt_client.post(
            f"/api/projects/{str(uuid4())}/dataset/media",
            files={"file": (f"test_file.{image_format}", BytesIO(b"123"), "image/jpeg")},
        )

        assert response.status_code == status.HTTP_201_CREATED
        assert response.json() == {
            "type": "image",
            "format": "jpg",
            "height": 768,
            "id": str(fxt_image_media.id),
            "name": "test_image",
            "size": 2048,
            "source_id": str(fxt_image_media.source_id),
            "width": 1024,
        }
        fxt_media_service.create_image.assert_called_once()
        metadata: ImageMetadata = fxt_media_service.create_image.call_args.args[0]
        assert metadata.project_id == fxt_get_project.id
        assert metadata.name == "test_file"
        assert metadata.image_format == ImageFormat(image_format)
        assert metadata.data
        fxt_dataset_service.create_dataset_item.assert_called_once_with(
            project_id=fxt_get_project.id,
            task=fxt_get_project.task,
            media=fxt_image_media,
            user_reviewed=False,
        )

    @pytest.mark.parametrize("image_format", ["svg"])
    def test_create_image_unsupported_image_format(
        self,
        fxt_get_project,
        fxt_media_service,
        fxt_dataset_service,
        fxt_client,
        image_format,
    ):
        response = fxt_client.post(
            f"/api/projects/{str(uuid4())}/dataset/media",
            files={"file": (f"test_file.{image_format}", BytesIO(b"123"), "image/jpeg")},
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
        fxt_media_service.create_image.assert_not_called()
        fxt_dataset_service.create_dataset_item.assert_not_called()

    def test_create_video_success(
        self, fxt_get_project, fxt_video_media, fxt_media_service, fxt_dataset_service, fxt_client
    ):
        fxt_media_service.create_video.return_value = fxt_video_media

        response = fxt_client.post(
            f"/api/projects/{str(uuid4())}/dataset/media",
            files={"file": ("test_file.mp4", BytesIO(b"123"), "video/mp4")},
        )

        assert response.status_code == status.HTTP_201_CREATED
        assert response.json() == {
            "type": "video",
            "format": "mp4",
            "height": 768,
            "id": str(fxt_video_media.id),
            "name": "test_video",
            "size": 2048,
            "source_id": str(fxt_video_media.source_id),
            "width": 1024,
            "fps": 25.0,
            "frame_count": 1000,
            "annotated_frame_count": 100,
            "duration": 40,
        }
        fxt_media_service.create_video.assert_called_once_with(
            project_id=fxt_get_project.id,
            data=ANY,
            name="test_file",
            video_format="mp4",
        )
        fxt_dataset_service.create_dataset_item.assert_not_called()

    def test_list_media(self, fxt_get_project, fxt_image_media, fxt_video_media, fxt_media_service, fxt_client):
        fxt_media_service.count_media.return_value = 2
        fxt_media_service.list_media.return_value = [fxt_image_media, fxt_video_media]

        response = fxt_client.get(f"/api/projects/{str(uuid4())}/dataset/media")

        assert response.status_code == status.HTTP_200_OK
        fxt_media_service.count_media.assert_called_once_with(
            project=fxt_get_project,
            start_date=None,
            end_date=None,
            annotation_status=None,
            label_ids=None,
            subsets=None,
            exclude_types=[MediaType.VIDEO_FRAME],
        )
        fxt_media_service.list_media.assert_called_once_with(
            project_id=fxt_get_project.id,
            filters=MediaFilters(
                limit=10, offset=0, start_date=None, end_date=None, annotation_status=None, label_ids=None, subsets=None
            ),
            exclude_types=[MediaType.VIDEO_FRAME],
        )

    def test_list_media_with_dataset_view_id(
        self,
        fxt_get_project,
        fxt_image_media,
        fxt_media_service,
        fxt_dataset_view_service,
        fxt_client,
    ):
        """When dataset_view_id is provided, the view-scoped service is used instead of the main media service."""
        dataset_view_id = uuid4()
        fxt_dataset_view_service.count_dataset_view_media.return_value = 1
        fxt_dataset_view_service.list_dataset_view_media.return_value = [fxt_image_media]

        response = fxt_client.get(f"/api/projects/{fxt_get_project.id}/dataset/media?dataset_view_id={dataset_view_id}")

        assert response.status_code == status.HTTP_200_OK
        fxt_dataset_view_service.count_dataset_view_media.assert_called_once_with(
            project_id=fxt_get_project.id,
            dataset_view_id=dataset_view_id,
            filters=MediaFilters(
                limit=10, offset=0, start_date=None, end_date=None, annotation_status=None, label_ids=None, subsets=None
            ),
        )
        fxt_dataset_view_service.list_dataset_view_media.assert_called_once_with(
            project_id=fxt_get_project.id,
            dataset_view_id=dataset_view_id,
            filters=MediaFilters(
                limit=10, offset=0, start_date=None, end_date=None, annotation_status=None, label_ids=None, subsets=None
            ),
        )
        fxt_media_service.count_media.assert_not_called()
        fxt_media_service.list_media.assert_not_called()

    def test_list_media_filtering_and_pagination(
        self, fxt_get_project, fxt_image_media, fxt_video_media, fxt_media_service, fxt_client
    ):
        fxt_media_service.count_media.return_value = 2
        fxt_media_service.list_media.return_value = [fxt_image_media, fxt_video_media]

        response = fxt_client.get(
            f"/api/projects/{str(uuid4())}/dataset/media?limit=50&offset=2&start_date=2025-01-09T00:00:00Z&end_date=2025-12-31T23:59:59Z"
        )

        assert response.status_code == status.HTTP_200_OK
        fxt_media_service.count_media.assert_called_once_with(
            project=fxt_get_project,
            start_date=datetime(2025, 1, 9, 0, 0, 0, tzinfo=ZoneInfo("UTC")),
            end_date=datetime(2025, 12, 31, 23, 59, 59, tzinfo=ZoneInfo("UTC")),
            annotation_status=None,
            label_ids=None,
            subsets=None,
            exclude_types=[MediaType.VIDEO_FRAME],
        )
        fxt_media_service.list_media.assert_called_once_with(
            project_id=fxt_get_project.id,
            filters=MediaFilters(
                limit=50,
                offset=2,
                start_date=datetime(2025, 1, 9, 0, 0, 0, tzinfo=ZoneInfo("UTC")),
                end_date=datetime(2025, 12, 31, 23, 59, 59, tzinfo=ZoneInfo("UTC")),
                annotation_status=None,
                label_ids=None,
                subsets=None,
            ),
            exclude_types=[MediaType.VIDEO_FRAME],
        )

    def test_list_media_naive_dates_are_normalized_to_utc(
        self, fxt_get_project, fxt_image_media, fxt_video_media, fxt_media_service, fxt_client
    ):
        fxt_media_service.count_media.return_value = 2
        fxt_media_service.list_media.return_value = [fxt_image_media, fxt_video_media]

        response = fxt_client.get(
            f"/api/projects/{str(uuid4())}/dataset/media?start_date=2025-01-09T00:00:00&end_date=2025-12-31T23:59:59"
        )

        assert response.status_code == status.HTTP_200_OK
        fxt_media_service.count_media.assert_called_once_with(
            project=fxt_get_project,
            start_date=datetime(2025, 1, 9, 0, 0, 0, tzinfo=UTC),
            end_date=datetime(2025, 12, 31, 23, 59, 59, tzinfo=UTC),
            annotation_status=None,
            label_ids=None,
            subsets=None,
            exclude_types=[MediaType.VIDEO_FRAME],
        )
        fxt_media_service.list_media.assert_called_once_with(
            project_id=fxt_get_project.id,
            filters=MediaFilters(
                limit=10,
                offset=0,
                start_date=datetime(2025, 1, 9, 0, 0, 0, tzinfo=UTC),
                end_date=datetime(2025, 12, 31, 23, 59, 59, tzinfo=UTC),
                annotation_status=None,
                label_ids=None,
                subsets=None,
            ),
            exclude_types=[MediaType.VIDEO_FRAME],
        )

    @pytest.mark.parametrize("limit", [1000, 0, -20])
    def test_list_media_wrong_limit(self, fxt_get_project, fxt_media_service, fxt_client, limit):
        response = fxt_client.get(f"/api/projects/{uuid4()}/dataset/media?limit=${limit}")

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
        fxt_media_service.list_media.assert_not_called()

    @pytest.mark.parametrize("offset", [-20])
    def test_list_media_wrong_offset(self, fxt_get_project, fxt_media_service, fxt_client, offset):
        response = fxt_client.get(f"/api/projects/{uuid4()}/dataset/media?offset=${offset}")

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
        fxt_media_service.list_media.assert_not_called()

    def test_list_media_offset_overflow(self, fxt_get_project, fxt_media_service, fxt_client):
        """Integers larger than 2^31-1 must be rejected with 422, not cause a SQLite overflow 500."""
        huge_offset = 2**63  # exceeds the le=2_147_483_647 bound
        response = fxt_client.get(f"/api/projects/{uuid4()}/dataset/media?offset={huge_offset}")

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
        fxt_media_service.list_media.assert_not_called()

    @pytest.mark.parametrize("offset", [-20])
    def test_list_media_wrong_dates(self, fxt_get_project, fxt_media_service, fxt_client, offset):
        response = fxt_client.get(
            f"/api/projects/{str(uuid4())}/dataset/media?start_date=2025-12-31T23:59:59Z&end_date=2025-01-09T00:00:00Z"
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
        fxt_media_service.list_media.assert_not_called()

    @pytest.mark.parametrize(
        "annotation_status",
        [
            DatasetItemAnnotationStatus.MISSING_ANNOTATIONS,
            DatasetItemAnnotationStatus.WITH_ANNOTATIONS,
        ],
    )
    def test_list_media_with_annotation_status(
        self, fxt_get_project, fxt_image_media, fxt_video_media, fxt_media_service, fxt_client, annotation_status
    ):
        fxt_media_service.count_media.return_value = 2
        fxt_media_service.list_media.return_value = [fxt_image_media, fxt_video_media]

        response = fxt_client.get(f"/api/projects/{str(uuid4())}/dataset/media?annotation_status={annotation_status}")

        assert response.status_code == status.HTTP_200_OK
        fxt_media_service.count_media.assert_called_once_with(
            project=fxt_get_project,
            start_date=None,
            end_date=None,
            annotation_status=annotation_status,
            label_ids=None,
            subsets=None,
            exclude_types=[MediaType.VIDEO_FRAME],
        )
        fxt_media_service.list_media.assert_called_once_with(
            project_id=fxt_get_project.id,
            filters=MediaFilters(
                limit=10,
                offset=0,
                start_date=None,
                end_date=None,
                annotation_status=annotation_status,
                label_ids=None,
                subsets=None,
            ),
            exclude_types=[MediaType.VIDEO_FRAME],
        )

    @pytest.mark.parametrize("subset", ["unassigned", "training", "validation", "testing"])
    def test_list_media_with_subset(
        self, fxt_get_project, fxt_image_media, fxt_video_media, fxt_media_service, fxt_client, subset
    ):
        fxt_media_service.count_media.return_value = 2
        fxt_media_service.list_media.return_value = [fxt_image_media, fxt_video_media]

        response = fxt_client.get(f"/api/projects/{str(uuid4())}/dataset/media?subsets={subset}")

        assert response.status_code == status.HTTP_200_OK
        fxt_media_service.count_media.assert_called_once_with(
            project=fxt_get_project,
            start_date=None,
            end_date=None,
            annotation_status=None,
            label_ids=None,
            subsets=[subset],
            exclude_types=[MediaType.VIDEO_FRAME],
        )
        fxt_media_service.list_media.assert_called_once_with(
            project_id=fxt_get_project.id,
            filters=MediaFilters(
                limit=10,
                offset=0,
                start_date=None,
                end_date=None,
                annotation_status=None,
                label_ids=None,
                subsets=[subset],
            ),
            exclude_types=[MediaType.VIDEO_FRAME],
        )

    def test_list_media_with_multiple_subsets(
        self, fxt_get_project, fxt_image_media, fxt_video_media, fxt_media_service, fxt_client
    ):
        fxt_media_service.count_media.return_value = 2
        fxt_media_service.list_media.return_value = [fxt_image_media, fxt_video_media]

        response = fxt_client.get(f"/api/projects/{str(uuid4())}/dataset/media?subsets=training&subsets=validation")

        assert response.status_code == status.HTTP_200_OK
        fxt_media_service.count_media.assert_called_once_with(
            project=fxt_get_project,
            start_date=None,
            end_date=None,
            annotation_status=None,
            label_ids=None,
            subsets=["training", "validation"],
            exclude_types=[MediaType.VIDEO_FRAME],
        )
        fxt_media_service.list_media.assert_called_once_with(
            project_id=fxt_get_project.id,
            filters=MediaFilters(
                limit=10,
                offset=0,
                start_date=None,
                end_date=None,
                annotation_status=None,
                label_ids=None,
                subsets=["training", "validation"],
            ),
            exclude_types=[MediaType.VIDEO_FRAME],
        )

    def test_list_media_default_sort_direction(
        self, fxt_get_project, fxt_image_media, fxt_video_media, fxt_media_service, fxt_client
    ):
        fxt_media_service.count_media.return_value = 2
        fxt_media_service.list_media.return_value = [fxt_image_media, fxt_video_media]

        response = fxt_client.get(f"/api/projects/{str(uuid4())}/dataset/media")

        assert response.status_code == status.HTTP_200_OK
        fxt_media_service.list_media.assert_called_once_with(
            project_id=fxt_get_project.id,
            filters=MediaFilters(
                limit=10,
                offset=0,
                start_date=None,
                end_date=None,
                annotation_status=None,
                label_ids=None,
                subsets=None,
                sort_by=MediaSortBy.UPLOAD_DATE,
                sort_direction=SortDirection.DESC,
            ),
            exclude_types=[MediaType.VIDEO_FRAME],
        )

    @pytest.mark.parametrize("sort_direction", [SortDirection.ASC, SortDirection.DESC])
    def test_list_media_with_sort_direction(
        self, fxt_get_project, fxt_image_media, fxt_video_media, fxt_media_service, fxt_client, sort_direction
    ):
        fxt_media_service.count_media.return_value = 2
        fxt_media_service.list_media.return_value = [fxt_image_media, fxt_video_media]

        response = fxt_client.get(f"/api/projects/{str(uuid4())}/dataset/media?sort_direction={sort_direction.value}")

        assert response.status_code == status.HTTP_200_OK
        fxt_media_service.list_media.assert_called_once_with(
            project_id=fxt_get_project.id,
            filters=MediaFilters(
                limit=10,
                offset=0,
                start_date=None,
                end_date=None,
                annotation_status=None,
                label_ids=None,
                subsets=None,
                sort_by=MediaSortBy.UPLOAD_DATE,
                sort_direction=sort_direction,
            ),
            exclude_types=[MediaType.VIDEO_FRAME],
        )

    @pytest.mark.parametrize("sort_by", list(MediaSortBy))
    def test_list_media_with_sort_by(
        self, fxt_get_project, fxt_image_media, fxt_video_media, fxt_media_service, fxt_client, sort_by
    ):
        fxt_media_service.count_media.return_value = 2
        fxt_media_service.list_media.return_value = [fxt_image_media, fxt_video_media]

        response = fxt_client.get(f"/api/projects/{str(uuid4())}/dataset/media?sort_by={sort_by.value}")

        assert response.status_code == status.HTTP_200_OK
        fxt_media_service.list_media.assert_called_once_with(
            project_id=fxt_get_project.id,
            filters=MediaFilters(
                limit=10,
                offset=0,
                start_date=None,
                end_date=None,
                annotation_status=None,
                label_ids=None,
                subsets=None,
                sort_by=sort_by,
                sort_direction=SortDirection.DESC,
            ),
            exclude_types=[MediaType.VIDEO_FRAME],
        )

    def test_list_media_wrong_sort_by(self, fxt_get_project, fxt_media_service, fxt_client):
        response = fxt_client.get(f"/api/projects/{uuid4()}/dataset/media?sort_by=bogus")

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
        fxt_media_service.list_media.assert_not_called()

    def test_list_media_wrong_sort_direction(self, fxt_get_project, fxt_media_service, fxt_client):
        response = fxt_client.get(f"/api/projects/{uuid4()}/dataset/media?sort_direction=bogus")

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
        fxt_media_service.list_media.assert_not_called()

    @pytest.mark.parametrize(
        "http_method, http_path, service_name, service_method",
        [
            ("get", f"/api/projects/{uuid4()}/dataset/media/invalid-id", "fxt_media_service", "get_media_by_id"),
            (
                "get",
                f"/api/projects/{uuid4()}/dataset/media/invalid-id/binary",
                "fxt_media_service",
                "get_media_binary_path_by_id",
            ),
            (
                "get",
                f"/api/projects/{uuid4()}/dataset/media/invalid-id/thumbnail",
                "fxt_media_service",
                "get_media_thumbnail_path",
            ),
            ("delete", f"/api/projects/{uuid4()}/dataset/media/invalid-id", "fxt_media_service", "delete_media"),
            (
                "post",
                f"/api/projects/{uuid4()}/dataset/media/invalid-id/annotations",
                "fxt_dataset_service",
                "set_dataset_item_annotations",
            ),
            (
                "get",
                f"/api/projects/{uuid4()}/dataset/media/invalid-id/annotations",
                "fxt_dataset_service",
                "get_dataset_item_by_id",
            ),
        ],
    )
    def test_invalid_ids(
        self, request, http_method, http_path, service_name, service_method, fxt_get_project, fxt_client
    ):
        service = request.getfixturevalue(service_name)
        response = getattr(fxt_client, http_method)(http_path)

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        getattr(service, service_method).assert_not_called()

    def test_get_media_not_found(self, fxt_get_project, fxt_media_service, fxt_client):
        media_id = uuid4()
        fxt_media_service.get_media_by_id.side_effect = ResourceNotFoundError(ResourceType.MEDIA, str(media_id))

        response = fxt_client.get(f"/api/projects/{str(uuid4())}/dataset/media/{str(media_id)}")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_media_service.get_media_by_id.assert_called_once_with(project_id=fxt_get_project.id, media_id=media_id)

    def test_get_image_success(self, fxt_image_media, fxt_get_project, fxt_media_service, fxt_client):
        fxt_media_service.get_media_by_id.return_value = fxt_image_media

        response = fxt_client.get(f"/api/projects/{str(uuid4())}/dataset/media/{str(fxt_image_media.id)}")

        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {
            "type": fxt_image_media.type,
            "format": fxt_image_media.format,
            "height": fxt_image_media.height,
            "id": str(fxt_image_media.id),
            "name": fxt_image_media.name,
            "size": fxt_image_media.size,
            "source_id": str(fxt_image_media.source_id),
            "width": fxt_image_media.width,
        }
        fxt_media_service.get_media_by_id.assert_called_once_with(
            project_id=fxt_get_project.id, media_id=fxt_image_media.id
        )

    def test_get_video_success(self, fxt_video_media, fxt_get_project, fxt_media_service, fxt_client):
        fxt_media_service.get_media_by_id.return_value = fxt_video_media

        response = fxt_client.get(f"/api/projects/{str(uuid4())}/dataset/media/{str(fxt_video_media.id)}")

        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {
            "type": fxt_video_media.type,
            "format": fxt_video_media.format,
            "height": fxt_video_media.height,
            "id": str(fxt_video_media.id),
            "name": fxt_video_media.name,
            "size": fxt_video_media.size,
            "source_id": str(fxt_video_media.source_id),
            "width": fxt_video_media.width,
            "fps": fxt_video_media.fps,
            "frame_count": fxt_video_media.frame_count,
            "annotated_frame_count": fxt_video_media.annotated_frame_count,
            "duration": fxt_video_media.duration,
        }
        fxt_media_service.get_media_by_id.assert_called_once_with(
            project_id=fxt_get_project.id, media_id=fxt_video_media.id
        )

    def test_get_video_frame_success(self, fxt_video_frame_media, fxt_get_project, fxt_media_service, fxt_client):
        fxt_media_service.get_media_by_id.return_value = fxt_video_frame_media

        response = fxt_client.get(f"/api/projects/{str(uuid4())}/dataset/media/{str(fxt_video_frame_media.id)}")

        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {
            "type": fxt_video_frame_media.type,
            "format": fxt_video_frame_media.format,
            "height": fxt_video_frame_media.height,
            "id": str(fxt_video_frame_media.id),
            "name": fxt_video_frame_media.name,
            "size": fxt_video_frame_media.size,
            "source_id": str(fxt_video_frame_media.source_id),
            "width": fxt_video_frame_media.width,
            "video_id": str(fxt_video_frame_media.video_id),
            "frame_index": fxt_video_frame_media.frame_index,
        }
        fxt_media_service.get_media_by_id.assert_called_once_with(
            project_id=fxt_get_project.id, media_id=fxt_video_frame_media.id
        )

    def test_get_media_binary_not_found(self, fxt_get_project, fxt_media_service, fxt_client):
        media_id = uuid4()
        fxt_media_service.get_media_by_id.side_effect = ResourceNotFoundError(ResourceType.MEDIA, str(media_id))

        response = fxt_client.get(f"/api/projects/{str(uuid4())}/dataset/media/{str(media_id)}/binary")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_media_service.get_media_by_id.assert_called_once_with(project_id=fxt_get_project.id, media_id=media_id)

    @pytest.mark.parametrize(
        "spec, media_type, format, suffix",
        [
            (Image, MediaType.IMAGE, ImageFormat.JPG, ".jpg"),
            (Video, MediaType.VIDEO, VideoFormat.MP4, ".mp4"),
            (VideoFrame, MediaType.VIDEO_FRAME, ImageFormat.JPG, ".jpg"),
        ],
    )
    def test_get_media_binary_success(
        self, fxt_get_project, fxt_media_service, fxt_client, spec, media_type, format, suffix
    ):
        media = MagicMock(spec=spec, id=uuid4(), format=format, type=media_type)
        type(media).name = PropertyMock(return_value="test")
        fxt_media_service.get_media_by_id.return_value = media

        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
                temp_file_path = Path(tmp_file.name)
                fxt_media_service.get_media_binary_path_by_id.return_value = temp_file_path
                # needs_display_normalization is tested separately; mock it out so that
                # arbitrary (possibly non-image) temp files don't cause PIL errors here.
                with patch("app.api.routers.media.needs_display_normalization", return_value=False):
                    response = fxt_client.get(f"/api/projects/{str(uuid4())}/dataset/media/{str(media.id)}/binary")

        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        assert response.status_code == status.HTTP_200_OK

        fxt_media_service.get_media_by_id.assert_called_once_with(project_id=fxt_get_project.id, media_id=media.id)
        fxt_media_service.get_media_binary_path_by_id.assert_called_once_with(
            project_id=fxt_get_project.id, media_id=media.id
        )

    @pytest.mark.parametrize(
        "image_format, suffix",
        [
            (ImageFormat.TIF, ".tif"),
            (ImageFormat.PNG, ".png"),
        ],
    )
    def test_get_media_binary_normalizes_high_bit_depth(
        self, fxt_get_project, fxt_media_service, fxt_client, image_format, suffix
    ):
        """High bit depth images are normalized to 8-bit PNG on the fly, regardless of format."""
        import numpy as np

        media = MagicMock(spec=Image, id=uuid4(), format=image_format, type=MediaType.IMAGE)
        type(media).name = PropertyMock(return_value="test_16bit")
        fxt_media_service.get_media_by_id.return_value = media

        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
                temp_file_path = Path(tmp_file.name)
                img = PILImage.fromarray(np.ones((10, 10), dtype="uint16") * 1000, mode="I;16")
                img.save(temp_file_path)
                fxt_media_service.get_media_binary_path_by_id.return_value = temp_file_path
                response = fxt_client.get(f"/api/projects/{str(uuid4())}/dataset/media/{str(media.id)}/binary")
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

        assert response.status_code == status.HTTP_200_OK
        assert "image/png" in response.headers.get("content-type", "")

    @pytest.mark.parametrize(
        "image_format, suffix",
        [
            (ImageFormat.TIF, ".tif"),
            (ImageFormat.PNG, ".png"),
        ],
    )
    def test_get_media_binary_raw_skips_normalization(
        self, fxt_get_project, fxt_media_service, fxt_client, image_format, suffix
    ):
        """When raw=true, the normalization function is never called regardless of format."""
        import numpy as np

        media = MagicMock(spec=Image, id=uuid4(), format=image_format, type=MediaType.IMAGE)
        type(media).name = PropertyMock(return_value="test_16bit")
        fxt_media_service.get_media_by_id.return_value = media

        temp_file_path = None
        try:
            with (
                tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file,
                patch("app.api.routers.media.normalize_image_to_png_bytes") as mock_normalize_image,
            ):
                temp_file_path = Path(tmp_file.name)
                img = PILImage.fromarray(np.ones((10, 10), dtype="uint16") * 1000, mode="I;16")
                img.save(temp_file_path)
                fxt_media_service.get_media_binary_path_by_id.return_value = temp_file_path
                response = fxt_client.get(f"/api/projects/{str(uuid4())}/dataset/media/{str(media.id)}/binary?raw=true")
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

        assert response.status_code == status.HTTP_200_OK
        mock_normalize_image.assert_not_called()

    def test_get_video_frame_binary_on_the_fly_annotated(self, fxt_get_project, fxt_media_service, fxt_client):
        video_id = uuid4()
        video_frame_id = uuid4()

        media = MagicMock(spec=Video, id=video_id, format=VideoFormat.MP4, type=MediaType.VIDEO, frame_count=100)
        fxt_media_service.get_media_by_id.return_value = media

        video_frame = MagicMock(spec=VideoFrame, id=video_frame_id, format=ImageFormat.JPG, type=MediaType.VIDEO_FRAME)
        type(video_frame).name = PropertyMock(return_value="test_10")
        fxt_media_service.get_video_frame_by_video_id_and_index.return_value = video_frame

        temp_file_path = None
        try:
            with (
                tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file,
                patch("app.api.routers.media.needs_display_normalization", return_value=False),
            ):
                temp_file_path = Path(tmp_file.name)
                fxt_media_service.get_media_binary_path_by_id.return_value = temp_file_path
                response = fxt_client.get(
                    f"/api/projects/{str(uuid4())}/dataset/media/{str(video_id)}/binary?frame_index=10"
                )
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        assert response.status_code == status.HTTP_200_OK

        fxt_media_service.get_media_by_id.assert_called_once_with(project_id=fxt_get_project.id, media_id=video_id)
        fxt_media_service.get_video_frame_by_video_id_and_index.assert_called_once_with(
            project=fxt_get_project, video_id=video_id, frame_index=10
        )
        fxt_media_service.get_media_binary_path_by_id.assert_called_once_with(
            project_id=fxt_get_project.id, media_id=video_frame_id
        )

    def test_get_video_frame_binary_on_the_fly_not_annotated(self, fxt_get_project, fxt_media_service, fxt_client):
        from PIL import Image

        video_id = uuid4()

        media = MagicMock(spec=Video, id=video_id, format=VideoFormat.MP4, type=MediaType.VIDEO, frame_count=100)
        type(media).name = PropertyMock(return_value="test")
        fxt_media_service.get_media_by_id.return_value = media

        fxt_media_service.get_video_frame_by_video_id_and_index.return_value = None
        test_image = Image.new("RGB", (64, 64), color="blue")
        fxt_media_service.get_frame_binary.return_value = test_image

        response = fxt_client.get(f"/api/projects/{str(uuid4())}/dataset/media/{str(video_id)}/binary?frame_index=10")
        assert response.status_code == status.HTTP_200_OK

        fxt_media_service.get_media_by_id.assert_called_once_with(project_id=fxt_get_project.id, media_id=video_id)
        fxt_media_service.get_video_frame_by_video_id_and_index.assert_called_once_with(
            project=fxt_get_project, video_id=video_id, frame_index=10
        )
        fxt_media_service.get_frame_binary.assert_called_once_with(
            project_id=fxt_get_project.id, video=media, frame_index=10
        )
        fxt_media_service.get_media_binary_path_by_id.assert_not_called()

    def test_get_video_frame_binary_on_the_fly_index_exceeds(self, fxt_get_project, fxt_media_service, fxt_client):
        video_id = uuid4()

        media = MagicMock(spec=Video, id=video_id, format=VideoFormat.MP4, type=MediaType.VIDEO, frame_count=10)
        type(media).name = PropertyMock(return_value="test")
        fxt_media_service.get_media_by_id.return_value = media

        response = fxt_client.get(f"/api/projects/{str(uuid4())}/dataset/media/{str(video_id)}/binary?frame_index=100")
        assert response.status_code == status.HTTP_400_BAD_REQUEST

        fxt_media_service.get_media_by_id.assert_called_once_with(project_id=fxt_get_project.id, media_id=video_id)
        fxt_media_service.get_video_frame_by_video_id_and_index.assert_not_called()
        fxt_media_service.get_frame_binary.assert_not_called()
        fxt_media_service.get_media_binary_path_by_id.assert_not_called()

    @pytest.mark.parametrize(
        "media",
        [
            (MagicMock(spec=Image, id=uuid4(), format=ImageFormat.JPG, type=MediaType.IMAGE)),
            (MagicMock(spec=Video, id=uuid4(), format=VideoFormat.MP4, type=MediaType.VIDEO)),
            (MagicMock(spec=VideoFrame, id=uuid4(), format=ImageFormat.JPG, type=MediaType.VIDEO_FRAME)),
        ],
    )
    def test_get_media_thumbnail_not_found(self, fxt_project_service, fxt_media_service, fxt_client, media):
        project_id = uuid4()
        media_id = uuid4()
        fxt_media_service.get_media_thumbnail_path_by_id.return_value = Path("non_existent")
        fxt_media_service.get_media_by_id.return_value = media

        response = fxt_client.get(f"/api/projects/{str(project_id)}/dataset/media/{str(media_id)}/thumbnail")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_media_service.get_media_thumbnail_path_by_id.assert_called_once_with(
            project_id=project_id, media_id=media_id
        )
        fxt_media_service.get_media_by_id.assert_called_once_with(project_id=project_id, media_id=media_id)

    def test_get_pregenerated_thumbnail_success(self, fxt_media_service, fxt_client):
        # Create a temporary JPEG file to act as the thumbnail
        project_id = uuid4()
        media_id = uuid4()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            thumbnail_path = Path(tmp_file.name)
            img = PILImage.new("RGB", (64, 64), color="red")
            img.save(tmp_file, format="JPEG")

        try:
            fxt_media_service.get_media_thumbnail_path_by_id.return_value = thumbnail_path

            response = fxt_client.get(f"/api/projects/{project_id}/dataset/media/{media_id}/thumbnail")

            assert response.status_code == status.HTTP_200_OK
            assert response.headers["content-type"] == "image/jpeg"
            with open(thumbnail_path, "rb") as f:
                assert response.content == f.read()
            fxt_media_service.get_media_thumbnail_path_by_id.assert_called_once_with(
                project_id=project_id, media_id=media_id
            )
        finally:
            if thumbnail_path.exists():
                os.unlink(thumbnail_path)

    def test_get_video_frame_thumbnail_on_the_fly_annotated(self, fxt_project_service, fxt_media_service, fxt_client):
        # Create a temporary JPEG file to act as the thumbnail
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            thumbnail_path = Path(tmp_file.name)
            img = PILImage.new("RGB", (64, 64), color="red")
            img.save(tmp_file, format="JPEG")

        try:
            project_id = uuid4()
            video_id = uuid4()
            video_frame_id = uuid4()

            project = MagicMock()

            media = MagicMock(spec=Video, id=video_id, format=VideoFormat.MP4, type=MediaType.VIDEO, frame_count=100)
            fxt_media_service.get_media_by_id.return_value = media

            fxt_media_service.get_media_thumbnail_path_by_id.side_effect = (Path("non_existent"), thumbnail_path)

            video_frame = MagicMock(spec=VideoFrame, id=video_frame_id, format=ImageFormat.JPG)
            type(video_frame).name = PropertyMock(return_value="test_10")
            fxt_project_service.get_project_by_id.return_value = project
            fxt_media_service.get_video_frame_by_video_id_and_index.return_value = video_frame

            response = fxt_client.get(
                f"/api/projects/{project_id}/dataset/media/{str(video_id)}/thumbnail?frame_index=10"
            )
            assert response.status_code == status.HTTP_200_OK

            fxt_media_service.get_media_by_id.assert_called_once_with(project_id=project_id, media_id=video_id)
            fxt_media_service.get_video_frame_by_video_id_and_index.assert_called_once_with(
                project=project, video_id=video_id, frame_index=10
            )
            fxt_media_service.get_media_thumbnail_path_by_id.assert_has_calls(
                [
                    call(project_id=project_id, media_id=video_id),
                    call(project_id=project_id, media_id=video_frame_id),
                ]
            )
        finally:
            if thumbnail_path.exists():
                os.unlink(thumbnail_path)

    def test_get_video_frame_thumbnail_on_the_fly_not_annotated(
        self, fxt_project_service, fxt_media_service, fxt_client
    ):
        video_id = uuid4()
        project_id = uuid4()

        project = MagicMock()

        fxt_media_service.get_media_thumbnail_path_by_id.return_value = Path("non_existent")

        media = MagicMock(spec=Video, id=video_id, format=VideoFormat.MP4, type=MediaType.VIDEO, frame_count=100)
        type(media).name = PropertyMock(return_value="test")
        fxt_media_service.get_media_by_id.return_value = media
        fxt_project_service.get_project_by_id.return_value = project

        fxt_media_service.get_video_frame_by_video_id_and_index.return_value = None
        test_image = PILImage.new("RGB", (64, 64), color="blue")
        fxt_media_service.get_frame_thumbnail.return_value = test_image

        response = fxt_client.get(
            f"/api/projects/{str(project_id)}/dataset/media/{str(video_id)}/thumbnail?frame_index=10"
        )
        assert response.status_code == status.HTTP_200_OK

        fxt_media_service.get_media_by_id.assert_called_once_with(project_id=project_id, media_id=video_id)
        fxt_media_service.get_video_frame_by_video_id_and_index.assert_called_once_with(
            project=project, video_id=video_id, frame_index=10
        )
        fxt_media_service.get_frame_thumbnail.assert_called_once_with(project=project, video=media, frame_index=10)

    def test_get_video_frame_thumbnail_on_the_fly_index_exceeds(
        self, fxt_project_service, fxt_media_service, fxt_client
    ):
        video_id = uuid4()
        project_id = uuid4()

        fxt_media_service.get_media_thumbnail_path_by_id.return_value = Path("non_existent")

        media = MagicMock(spec=Video, id=video_id, format=VideoFormat.MP4, type=MediaType.VIDEO, frame_count=10)
        type(media).name = PropertyMock(return_value="test")
        fxt_media_service.get_media_by_id.return_value = media

        response = fxt_client.get(
            f"/api/projects/{str(project_id)}/dataset/media/{str(video_id)}/thumbnail?frame_index=100"
        )
        assert response.status_code == status.HTTP_400_BAD_REQUEST

        fxt_media_service.get_media_by_id.assert_called_once_with(project_id=project_id, media_id=video_id)
        fxt_media_service.get_video_frame_by_video_id_and_index.assert_not_called()
        fxt_media_service.get_frame_thumbnail.assert_not_called()

    def test_delete_media_not_found(self, fxt_get_project, fxt_media_service, fxt_client):
        media_id = uuid4()
        fxt_media_service.delete_media.side_effect = ResourceNotFoundError(ResourceType.DATASET_ITEM, str(media_id))

        response = fxt_client.delete(f"/api/projects/{str(uuid4())}/dataset/media/{str(media_id)}")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_media_service.delete_media.assert_called_once_with(project=fxt_get_project, media_id=media_id)

    def test_delete_media_success(self, fxt_get_project, fxt_media_service, fxt_client):
        media_id = uuid4()
        response = fxt_client.delete(f"/api/projects/{str(uuid4())}/dataset/media/{str(media_id)}")

        assert response.status_code == status.HTTP_204_NO_CONTENT
        fxt_media_service.delete_media.assert_called_once_with(project=fxt_get_project, media_id=media_id)

    def test_bulk_delete_media_skips_not_found(self, fxt_get_project, fxt_media_service, fxt_client):
        media_id_1 = uuid4()
        media_id_2 = uuid4()
        media_id_3 = uuid4()

        fxt_media_service.delete_media.side_effect = [
            None,
            ResourceNotFoundError(ResourceType.MEDIA, str(media_id_2)),
            None,
        ]

        response = fxt_client.request(
            "DELETE",
            f"/api/projects/{str(uuid4())}/dataset/media",
            json={"media_ids": [str(media_id_1), str(media_id_2), str(media_id_3)]},
        )

        assert response.status_code == status.HTTP_204_NO_CONTENT
        assert fxt_media_service.delete_media.call_count == 3
        fxt_media_service.delete_media.assert_has_calls(
            [
                call(project=fxt_get_project, media_id=media_id_1),
                call(project=fxt_get_project, media_id=media_id_2),
                call(project=fxt_get_project, media_id=media_id_3),
            ]
        )

    @pytest.mark.parametrize(
        "media",
        [
            MagicMock(spec=Image, type=MediaType.IMAGE, id=uuid4()),
            MagicMock(spec=VideoFrame, type=MediaType.VIDEO_FRAME, id=uuid4()),
        ],
    )
    def test_set_media_annotations_success(
        self, media, fxt_get_project, fxt_media_service, fxt_dataset_service, fxt_client
    ):
        label_id = uuid4()
        annotations = [
            DatasetItemAnnotation(
                labels=[LabelReference(id=label_id)],
                shape=Rectangle(type="rectangle", x=0, y=0, width=10, height=10),
            )
        ]
        fxt_media_service.get_media_by_id.return_value = media
        dataset_item = MagicMock(
            spec=DatasetItem,
            annotation_data=annotations,
            user_reviewed=True,
            prediction_model_id=None,
            subset=DatasetItemSubset.UNASSIGNED,
        )
        fxt_dataset_service.set_dataset_item_annotations.return_value = dataset_item

        response = fxt_client.post(
            f"/api/projects/{str(uuid4())}/dataset/media/{str(media.id)}/annotations",
            json=SetMediaAnnotations(annotations=annotations).model_dump(mode="json"),
        )

        assert response.status_code == status.HTTP_201_CREATED
        assert response.json() == {
            "media_id": str(media.id),
            "annotations": [
                {
                    "confidences": None,
                    "labels": [{"id": str(label_id)}],
                    "shape": {"height": 10, "type": "rectangle", "width": 10, "x": 0, "y": 0},
                }
            ],
            "prediction_model_id": None,
            "user_reviewed": True,
            "subset": "unassigned",
        }
        fxt_media_service.get_media_by_id.assert_called_once_with(project_id=fxt_get_project.id, media_id=media.id)
        fxt_dataset_service.set_dataset_item_annotations.assert_called_once_with(
            project=fxt_get_project,
            dataset_item_id=media.id,
            annotations=annotations,
            user_reviewed=True,
            prediction_model_id=None,
        )

    def test_set_video_annotations_missing_frame_index(
        self, fxt_get_project, fxt_media_service, fxt_dataset_service, fxt_client
    ):
        label_id = uuid4()
        annotations = [
            DatasetItemAnnotation(
                labels=[LabelReference(id=label_id)],
                shape=Rectangle(type="rectangle", x=0, y=0, width=10, height=10),
            )
        ]
        media = MagicMock(spec=Video, type=MediaType.VIDEO, id=uuid4())
        fxt_media_service.get_media_by_id.return_value = media

        response = fxt_client.post(
            f"/api/projects/{str(uuid4())}/dataset/media/{str(media.id)}/annotations",
            json=SetMediaAnnotations(annotations=annotations).model_dump(mode="json"),
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        fxt_media_service.get_media_by_id.assert_called_once_with(project_id=fxt_get_project.id, media_id=media.id)
        fxt_dataset_service.set_dataset_item_annotations.assert_not_called()

    def test_set_video_annotations_index_exceeds(
        self, fxt_get_project, fxt_media_service, fxt_dataset_service, fxt_client
    ):
        label_id = uuid4()
        annotations = [
            DatasetItemAnnotation(
                labels=[LabelReference(id=label_id)],
                shape=Rectangle(type="rectangle", x=0, y=0, width=10, height=10),
            )
        ]
        media = MagicMock(spec=Video, id=uuid4(), type=MediaType.VIDEO, frame_count=10)
        fxt_media_service.get_media_by_id.return_value = media

        response = fxt_client.post(
            f"/api/projects/{str(uuid4())}/dataset/media/{str(media.id)}/annotations?frame_index=100",
            json=SetMediaAnnotations(annotations=annotations).model_dump(mode="json"),
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        fxt_media_service.get_media_by_id.assert_called_once_with(project_id=fxt_get_project.id, media_id=media.id)
        fxt_dataset_service.set_dataset_item_annotations.assert_not_called()

    def test_set_video_annotations_existing_frame(
        self, fxt_get_project, fxt_media_service, fxt_dataset_service, fxt_client
    ):
        label_id = uuid4()
        video_frame_id = uuid4()
        annotations = [
            DatasetItemAnnotation(
                labels=[LabelReference(id=label_id)],
                shape=Rectangle(type="rectangle", x=0, y=0, width=10, height=10),
            )
        ]
        media = MagicMock(spec=Video, type=MediaType.VIDEO, frame_count=20, id=uuid4())
        fxt_media_service.get_media_by_id.return_value = media
        video_frame = MagicMock(
            spec=VideoFrame,
            id=video_frame_id,
        )
        fxt_media_service.get_video_frame_by_video_id_and_index.return_value = video_frame
        dataset_item = MagicMock(
            spec=DatasetItem,
            annotation_data=annotations,
            user_reviewed=True,
            prediction_model_id=None,
            subset=DatasetItemSubset.UNASSIGNED,
        )
        fxt_dataset_service.set_dataset_item_annotations.return_value = dataset_item

        response = fxt_client.post(
            f"/api/projects/{str(uuid4())}/dataset/media/{str(media.id)}/annotations?frame_index=10",
            json=SetMediaAnnotations(annotations=annotations).model_dump(mode="json"),
        )

        assert response.status_code == status.HTTP_201_CREATED
        assert response.json() == {
            "media_id": str(video_frame_id),
            "annotations": [
                {
                    "confidences": None,
                    "labels": [{"id": str(label_id)}],
                    "shape": {"height": 10, "type": "rectangle", "width": 10, "x": 0, "y": 0},
                }
            ],
            "prediction_model_id": None,
            "user_reviewed": True,
            "subset": "unassigned",
        }
        fxt_media_service.get_media_by_id.assert_called_once_with(project_id=fxt_get_project.id, media_id=media.id)
        fxt_media_service.get_video_frame_by_video_id_and_index.assert_called_once_with(
            project=fxt_get_project, video_id=media.id, frame_index=10
        )
        fxt_dataset_service.set_dataset_item_annotations.assert_called_once_with(
            project=fxt_get_project,
            dataset_item_id=video_frame_id,
            annotations=annotations,
            user_reviewed=True,
            prediction_model_id=None,
        )

    def test_set_video_annotations_extract_frame(
        self, fxt_get_project, fxt_media_service, fxt_dataset_service, fxt_client
    ):
        label_id = uuid4()
        video_frame_id = uuid4()
        annotations = [
            DatasetItemAnnotation(
                labels=[LabelReference(id=label_id)],
                shape=Rectangle(type="rectangle", x=0, y=0, width=10, height=10),
            )
        ]
        media = MagicMock(spec=Video, type=MediaType.VIDEO, frame_count=20, id=uuid4())
        fxt_media_service.get_media_by_id.return_value = media
        fxt_media_service.get_video_frame_by_video_id_and_index.return_value = None
        frame_binary = PILImage.new("RGB", (64, 64), color="red")
        fxt_media_service.get_frame_binary.return_value = frame_binary
        video_frame = MagicMock(
            spec=VideoFrame,
            id=video_frame_id,
            type=MediaType.VIDEO_FRAME,
        )
        fxt_media_service.save_video_frame.return_value = video_frame
        dataset_item = MagicMock(
            spec=DatasetItem,
            annotation_data=annotations,
            user_reviewed=True,
            prediction_model_id=None,
            subset=DatasetItemSubset.UNASSIGNED,
        )
        fxt_dataset_service.set_dataset_item_annotations.return_value = dataset_item

        response = fxt_client.post(
            f"/api/projects/{str(uuid4())}/dataset/media/{str(media.id)}/annotations?frame_index=10",
            json=SetMediaAnnotations(annotations=annotations).model_dump(mode="json"),
        )

        assert response.status_code == status.HTTP_201_CREATED
        assert response.json() == {
            "media_id": str(video_frame_id),
            "annotations": [
                {
                    "confidences": None,
                    "labels": [{"id": str(label_id)}],
                    "shape": {"height": 10, "type": "rectangle", "width": 10, "x": 0, "y": 0},
                }
            ],
            "prediction_model_id": None,
            "user_reviewed": True,
            "subset": "unassigned",
        }
        fxt_media_service.get_media_by_id.assert_called_once_with(project_id=fxt_get_project.id, media_id=media.id)
        fxt_media_service.get_video_frame_by_video_id_and_index.assert_called_once_with(
            project=fxt_get_project, video_id=media.id, frame_index=10
        )
        fxt_media_service.get_frame_binary.assert_called_once_with(
            project_id=fxt_get_project.id, video=media, frame_index=10
        )
        fxt_media_service.save_video_frame.assert_called_once_with(
            project=fxt_get_project, video=media, frame_index=10, frame_image=frame_binary
        )
        fxt_dataset_service.create_dataset_item.assert_called_once_with(
            project_id=fxt_get_project.id,
            task=fxt_get_project.task,
            media=video_frame,
            user_reviewed=True,
        )
        fxt_dataset_service.set_dataset_item_annotations.assert_called_once_with(
            project=fxt_get_project,
            dataset_item_id=video_frame_id,
            annotations=annotations,
            user_reviewed=True,
            prediction_model_id=None,
        )

    @pytest.mark.parametrize(
        "media",
        [
            MagicMock(spec=Image, type=MediaType.IMAGE, id=uuid4()),
            MagicMock(spec=VideoFrame, type=MediaType.VIDEO_FRAME, id=uuid4()),
        ],
    )
    def test_set_media_annotations_label_not_found(
        self, media, fxt_get_project, fxt_media_service, fxt_dataset_service, fxt_client
    ):
        label_id = uuid4()
        annotations = [
            DatasetItemAnnotation(
                labels=[LabelReference(id=label_id)],
                shape=Rectangle(type="rectangle", x=0, y=0, width=10, height=10),
            )
        ]
        fxt_media_service.get_media_by_id.return_value = media
        fxt_dataset_service.set_dataset_item_annotations.side_effect = AnnotationValidationError(str(label_id))

        response = fxt_client.post(
            f"/api/projects/{str(uuid4())}/dataset/media/{str(media.id)}/annotations",
            json=SetMediaAnnotations(annotations=annotations).model_dump(mode="json"),
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        fxt_dataset_service.set_dataset_item_annotations.assert_called_once_with(
            project=fxt_get_project,
            dataset_item_id=media.id,
            annotations=annotations,
            user_reviewed=True,
            prediction_model_id=None,
        )

    @pytest.mark.parametrize(
        "media",
        [
            MagicMock(spec=Image, type=MediaType.IMAGE, id=uuid4()),
            MagicMock(spec=VideoFrame, type=MediaType.VIDEO_FRAME, id=uuid4()),
        ],
    )
    def test_set_media_annotations_not_found(
        self, media, fxt_get_project, fxt_media_service, fxt_dataset_service, fxt_client
    ):
        label_id = uuid4()
        annotations = [
            DatasetItemAnnotation(
                labels=[LabelReference(id=label_id)],
                shape=Rectangle(type="rectangle", x=0, y=0, width=10, height=10),
            )
        ]
        fxt_media_service.get_media_by_id.return_value = media
        fxt_dataset_service.set_dataset_item_annotations.side_effect = ResourceNotFoundError(
            ResourceType.DATASET_ITEM, str(media.id)
        )

        response = fxt_client.post(
            f"/api/projects/{str(uuid4())}/dataset/media/{str(media.id)}/annotations",
            json=SetMediaAnnotations(annotations=annotations).model_dump(mode="json"),
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_dataset_service.set_dataset_item_annotations.assert_called_once_with(
            project=fxt_get_project,
            dataset_item_id=media.id,
            annotations=annotations,
            user_reviewed=True,
            prediction_model_id=None,
        )

    @pytest.mark.parametrize(
        "media",
        [
            MagicMock(spec=Image, type=MediaType.IMAGE, id=uuid4()),
            MagicMock(spec=VideoFrame, type=MediaType.VIDEO_FRAME, id=uuid4()),
        ],
    )
    def test_set_media_annotations_with_subset(
        self, media, fxt_get_project, fxt_media_service, fxt_dataset_service, fxt_client
    ):
        label_id = uuid4()
        annotations = [
            DatasetItemAnnotation(
                labels=[LabelReference(id=label_id)],
                shape=Rectangle(type="rectangle", x=0, y=0, width=10, height=10),
            )
        ]
        fxt_media_service.get_media_by_id.return_value = media
        dataset_item = MagicMock(
            spec=DatasetItem,
            annotation_data=annotations,
            user_reviewed=True,
            prediction_model_id=None,
            subset=DatasetItemSubset.UNASSIGNED,
        )
        fxt_dataset_service.set_dataset_item_annotations.return_value = dataset_item
        updated_dataset_item = MagicMock(
            spec=DatasetItem,
            annotation_data=annotations,
            user_reviewed=True,
            prediction_model_id=None,
            subset=DatasetItemSubset.TRAINING,
        )
        fxt_dataset_service.assign_dataset_item_subset.return_value = updated_dataset_item

        response = fxt_client.post(
            f"/api/projects/{str(uuid4())}/dataset/media/{str(media.id)}/annotations",
            json=SetMediaAnnotations(annotations=annotations, subset=DatasetItemSubset.TRAINING).model_dump(
                mode="json"
            ),
        )

        assert response.status_code == status.HTTP_201_CREATED
        assert response.json()["subset"] == "training"
        fxt_dataset_service.set_dataset_item_annotations.assert_called_once_with(
            project=fxt_get_project,
            dataset_item_id=media.id,
            annotations=annotations,
            user_reviewed=True,
            prediction_model_id=None,
        )
        fxt_dataset_service.assign_dataset_item_subset.assert_called_once_with(
            project_id=fxt_get_project.id,
            dataset_item_id=media.id,
            subset=DatasetItemSubset.TRAINING,
        )

    @pytest.mark.parametrize(
        "media",
        [
            MagicMock(spec=Image, type=MediaType.IMAGE, id=uuid4()),
            MagicMock(spec=VideoFrame, type=MediaType.VIDEO_FRAME, id=uuid4()),
        ],
    )
    def test_set_media_annotations_with_subset_already_assigned_different(
        self, media, fxt_get_project, fxt_media_service, fxt_dataset_service, fxt_client
    ):
        label_id = uuid4()
        annotations = [
            DatasetItemAnnotation(
                labels=[LabelReference(id=label_id)],
                shape=Rectangle(type="rectangle", x=0, y=0, width=10, height=10),
            )
        ]
        fxt_media_service.get_media_by_id.return_value = media
        dataset_item = MagicMock(
            spec=DatasetItem,
            annotation_data=annotations,
            user_reviewed=True,
            prediction_model_id=None,
            subset=DatasetItemSubset.TRAINING,
        )
        fxt_dataset_service.set_dataset_item_annotations.return_value = dataset_item
        fxt_dataset_service.assign_dataset_item_subset.side_effect = SubsetAlreadyAssignedError

        response = fxt_client.post(
            f"/api/projects/{str(uuid4())}/dataset/media/{str(media.id)}/annotations",
            json=SetMediaAnnotations(annotations=annotations, subset=DatasetItemSubset.VALIDATION).model_dump(
                mode="json"
            ),
        )

        assert response.status_code == status.HTTP_409_CONFLICT
        fxt_dataset_service.assign_dataset_item_subset.assert_called_once_with(
            project_id=fxt_get_project.id,
            dataset_item_id=media.id,
            subset=DatasetItemSubset.VALIDATION,
        )

    @pytest.mark.parametrize(
        "media",
        [
            MagicMock(spec=Image, type=MediaType.IMAGE, id=uuid4()),
            MagicMock(spec=VideoFrame, type=MediaType.VIDEO_FRAME, id=uuid4()),
        ],
    )
    def test_set_media_annotations_without_subset(
        self, media, fxt_get_project, fxt_media_service, fxt_dataset_service, fxt_client
    ):
        label_id = uuid4()
        annotations = [
            DatasetItemAnnotation(
                labels=[LabelReference(id=label_id)],
                shape=Rectangle(type="rectangle", x=0, y=0, width=10, height=10),
            )
        ]
        fxt_media_service.get_media_by_id.return_value = media
        dataset_item = MagicMock(
            spec=DatasetItem,
            annotation_data=annotations,
            user_reviewed=True,
            prediction_model_id=None,
            subset=DatasetItemSubset.UNASSIGNED,
        )
        fxt_dataset_service.set_dataset_item_annotations.return_value = dataset_item

        response = fxt_client.post(
            f"/api/projects/{str(uuid4())}/dataset/media/{str(media.id)}/annotations",
            json=SetMediaAnnotations(annotations=annotations).model_dump(mode="json"),
        )

        assert response.status_code == status.HTTP_201_CREATED
        fxt_dataset_service.assign_dataset_item_subset.assert_not_called()

    @pytest.mark.parametrize(
        "media",
        [
            MagicMock(spec=Image, type=MediaType.IMAGE, id=uuid4()),
            MagicMock(spec=VideoFrame, type=MediaType.VIDEO_FRAME, id=uuid4()),
        ],
    )
    def test_get_media_annotations(self, media, fxt_get_project, fxt_media_service, fxt_dataset_service, fxt_client):
        label_id = uuid4()
        fxt_media_service.get_media_by_id.return_value = media
        dataset_item = MagicMock(
            spec=DatasetItem,
            annotation_data=[
                DatasetItemAnnotation(
                    labels=[LabelReference(id=label_id)],
                    shape=Rectangle(type="rectangle", x=0, y=0, width=10, height=10),
                )
            ],
            user_reviewed=True,
            prediction_model_id=None,
            subset=DatasetItemSubset.UNASSIGNED,
        )
        fxt_dataset_service.get_dataset_item_by_id.return_value = dataset_item

        response = fxt_client.get(f"/api/projects/{str(uuid4())}/dataset/media/{str(media.id)}/annotations")

        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {
            "annotations": [
                {
                    "confidences": None,
                    "labels": [{"id": str(label_id)}],
                    "shape": {"height": 10, "type": "rectangle", "width": 10, "x": 0, "y": 0},
                }
            ],
            "prediction_model_id": None,
            "user_reviewed": True,
            "subset": "unassigned",
        }
        fxt_media_service.get_media_by_id.assert_called_once_with(project_id=fxt_get_project.id, media_id=media.id)
        fxt_dataset_service.get_dataset_item_by_id.assert_called_once_with(
            project_id=fxt_get_project.id,
            dataset_item_id=media.id,
        )

    def test_get_video_annotations_missing_frame_index(
        self, fxt_get_project, fxt_media_service, fxt_dataset_service, fxt_client
    ):
        media = MagicMock(spec=Video, type=MediaType.VIDEO, frame_count=20, id=uuid4())
        fxt_media_service.get_media_by_id.return_value = media

        response = fxt_client.get(f"/api/projects/{str(uuid4())}/dataset/media/{str(media.id)}/annotations")

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        fxt_dataset_service.get_dataset_item_by_id.assert_not_called()

    def test_get_video_annotations_index_exceeds(
        self, fxt_get_project, fxt_media_service, fxt_dataset_service, fxt_client
    ):
        media = MagicMock(spec=Video, type=MediaType.VIDEO, frame_count=10, id=uuid4())
        fxt_media_service.get_media_by_id.return_value = media

        response = fxt_client.get(
            f"/api/projects/{str(uuid4())}/dataset/media/{str(media.id)}/annotations?frame_index=100"
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        fxt_dataset_service.get_dataset_item_by_id.assert_not_called()

    def test_get_video_annotations_existing_frame(
        self, fxt_get_project, fxt_media_service, fxt_dataset_service, fxt_client
    ):
        label_id = uuid4()
        video_frame_id = uuid4()
        media = MagicMock(spec=Video, type=MediaType.VIDEO, frame_count=20, id=uuid4())
        fxt_media_service.get_media_by_id.return_value = media
        video_frame = MagicMock(
            spec=VideoFrame,
            id=video_frame_id,
        )
        fxt_media_service.get_video_frame_by_video_id_and_index.return_value = video_frame
        dataset_item = MagicMock(
            spec=DatasetItem,
            annotation_data=[
                DatasetItemAnnotation(
                    labels=[LabelReference(id=label_id)],
                    shape=Rectangle(type="rectangle", x=0, y=0, width=10, height=10),
                )
            ],
            user_reviewed=True,
            prediction_model_id=None,
            subset=DatasetItemSubset.UNASSIGNED,
        )
        fxt_dataset_service.get_dataset_item_by_id.return_value = dataset_item

        response = fxt_client.get(
            f"/api/projects/{str(uuid4())}/dataset/media/{str(media.id)}/annotations?frame_index=10"
        )

        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {
            "annotations": [
                {
                    "confidences": None,
                    "labels": [{"id": str(label_id)}],
                    "shape": {"height": 10, "type": "rectangle", "width": 10, "x": 0, "y": 0},
                }
            ],
            "prediction_model_id": None,
            "user_reviewed": True,
            "subset": "unassigned",
        }
        fxt_media_service.get_media_by_id.assert_called_once_with(project_id=fxt_get_project.id, media_id=media.id)
        fxt_media_service.get_video_frame_by_video_id_and_index.assert_called_once_with(
            project=fxt_get_project, video_id=media.id, frame_index=10
        )
        fxt_dataset_service.get_dataset_item_by_id.assert_called_once_with(
            project_id=fxt_get_project.id,
            dataset_item_id=video_frame_id,
        )

    def test_get_video_annotations_frame_not_found(
        self, fxt_get_project, fxt_media_service, fxt_dataset_service, fxt_client
    ):
        label_id = uuid4()
        media = MagicMock(spec=Video, type=MediaType.VIDEO, frame_count=20, id=uuid4())
        fxt_media_service.get_media_by_id.return_value = media
        fxt_media_service.get_video_frame_by_video_id_and_index.return_value = None
        dataset_item = MagicMock(
            spec=DatasetItem,
            annotation_data=[
                DatasetItemAnnotation(
                    labels=[LabelReference(id=label_id)],
                    shape=Rectangle(type="rectangle", x=0, y=0, width=10, height=10),
                )
            ],
            user_reviewed=True,
            prediction_model_id=None,
        )
        fxt_dataset_service.get_dataset_item_by_id.return_value = dataset_item

        response = fxt_client.get(
            f"/api/projects/{str(uuid4())}/dataset/media/{str(media.id)}/annotations?frame_index=10"
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_media_service.get_media_by_id.assert_called_once_with(project_id=fxt_get_project.id, media_id=media.id)
        fxt_media_service.get_video_frame_by_video_id_and_index.assert_called_once_with(
            project=fxt_get_project, video_id=media.id, frame_index=10
        )
        fxt_dataset_service.get_dataset_item_by_id.assert_not_called()

    @pytest.mark.parametrize(
        "media",
        [
            MagicMock(spec=Image, type=MediaType.IMAGE, id=uuid4()),
            MagicMock(spec=VideoFrame, type=MediaType.VIDEO_FRAME, id=uuid4()),
        ],
    )
    def test_get_media_annotations_not_found(
        self, media, fxt_get_project, fxt_media_service, fxt_dataset_service, fxt_client
    ):
        fxt_media_service.get_media_by_id.return_value = media
        fxt_dataset_service.get_dataset_item_by_id.side_effect = ResourceNotFoundError(
            ResourceType.DATASET_ITEM, str(media.id)
        )

        response = fxt_client.get(f"/api/projects/{str(uuid4())}/dataset/media/{str(media.id)}/annotations")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_media_service.get_media_by_id.assert_called_once_with(project_id=fxt_get_project.id, media_id=media.id)
        fxt_dataset_service.get_dataset_item_by_id.assert_called_once_with(
            project_id=fxt_get_project.id,
            dataset_item_id=media.id,
        )

    @pytest.mark.parametrize(
        "media",
        [
            MagicMock(spec=Image, type=MediaType.IMAGE, id=uuid4()),
            MagicMock(spec=VideoFrame, type=MediaType.VIDEO_FRAME, id=uuid4()),
        ],
    )
    def test_get_media_annotations_not_annotated(
        self, media, fxt_get_project, fxt_media_service, fxt_dataset_service, fxt_client
    ):
        fxt_media_service.get_media_by_id.return_value = media
        dataset_item = MagicMock(spec=DatasetItem, annotation_data=None, user_reviewed=False, prediction_model_id=None)
        fxt_dataset_service.get_dataset_item_by_id.return_value = dataset_item

        response = fxt_client.get(f"/api/projects/{str(uuid4())}/dataset/media/{str(media.id)}/annotations")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_media_service.get_media_by_id.assert_called_once_with(project_id=fxt_get_project.id, media_id=media.id)
        fxt_dataset_service.get_dataset_item_by_id.assert_called_once_with(
            project_id=fxt_get_project.id,
            dataset_item_id=media.id,
        )

    @pytest.mark.parametrize(
        "media",
        [
            MagicMock(spec=Image, type=MediaType.IMAGE, id=uuid4()),
            MagicMock(spec=VideoFrame, type=MediaType.VIDEO_FRAME, id=uuid4()),
        ],
    )
    def test_delete_media_annotations(self, media, fxt_get_project, fxt_media_service, fxt_dataset_service, fxt_client):
        fxt_media_service.get_media_by_id.return_value = media

        response = fxt_client.delete(f"/api/projects/{str(uuid4())}/dataset/media/{str(media.id)}/annotations")

        assert response.status_code == status.HTTP_204_NO_CONTENT
        fxt_media_service.get_media_by_id.assert_called_once_with(project_id=fxt_get_project.id, media_id=media.id)
        fxt_dataset_service.delete_dataset_item_annotations.assert_called_once_with(
            project=fxt_get_project,
            dataset_item_id=media.id,
        )

    def test_delete_video_annotations_missing_frame_index(
        self, fxt_get_project, fxt_media_service, fxt_dataset_service, fxt_client
    ):
        media = MagicMock(spec=Video, type=MediaType.VIDEO, id=uuid4())
        fxt_media_service.get_media_by_id.return_value = media

        response = fxt_client.delete(f"/api/projects/{str(uuid4())}/dataset/media/{str(media.id)}/annotations")

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        fxt_media_service.get_media_by_id.assert_called_once_with(project_id=fxt_get_project.id, media_id=media.id)
        fxt_dataset_service.delete_dataset_item_annotations.assert_not_called()

    def test_delete_video_annotations_index_exceeds(
        self, fxt_get_project, fxt_media_service, fxt_dataset_service, fxt_client
    ):
        media = MagicMock(spec=Video, type=MediaType.VIDEO, frame_count=10, id=uuid4())
        fxt_media_service.get_media_by_id.return_value = media

        response = fxt_client.delete(
            f"/api/projects/{str(uuid4())}/dataset/media/{str(media.id)}/annotations?frame_index=100"
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        fxt_media_service.get_media_by_id.assert_called_once_with(project_id=fxt_get_project.id, media_id=media.id)
        fxt_dataset_service.delete_dataset_item_annotations.assert_not_called()

    def test_delete_video_annotations_existing_frame(
        self, fxt_get_project, fxt_media_service, fxt_dataset_service, fxt_client
    ):
        video_frame_id = uuid4()
        media = MagicMock(spec=Video, type=MediaType.VIDEO, frame_count=20, id=uuid4())
        fxt_media_service.get_media_by_id.return_value = media
        video_frame = MagicMock(
            spec=VideoFrame,
            id=video_frame_id,
        )
        fxt_media_service.get_video_frame_by_video_id_and_index.return_value = video_frame

        response = fxt_client.delete(
            f"/api/projects/{str(uuid4())}/dataset/media/{str(media.id)}/annotations?frame_index=10"
        )

        assert response.status_code == status.HTTP_204_NO_CONTENT
        fxt_media_service.get_media_by_id.assert_called_once_with(project_id=fxt_get_project.id, media_id=media.id)
        fxt_dataset_service.delete_dataset_item_annotations.assert_called_once_with(
            project=fxt_get_project,
            dataset_item_id=video_frame_id,
        )

    def test_delete_video_annotations_frame_not_found(
        self, fxt_get_project, fxt_media_service, fxt_dataset_service, fxt_client
    ):
        media = MagicMock(spec=Video, type=MediaType.VIDEO, frame_count=20, id=uuid4())
        fxt_media_service.get_media_by_id.return_value = media
        fxt_media_service.get_video_frame_by_video_id_and_index.return_value = None

        response = fxt_client.delete(
            f"/api/projects/{str(uuid4())}/dataset/media/{str(media.id)}/annotations?frame_index=10"
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_media_service.get_media_by_id.assert_called_once_with(project_id=fxt_get_project.id, media_id=media.id)
        fxt_dataset_service.delete_dataset_item_annotations.assert_not_called()

    @pytest.mark.parametrize(
        "media",
        [
            MagicMock(spec=Image, type=MediaType.IMAGE, id=uuid4()),
            MagicMock(spec=VideoFrame, type=MediaType.VIDEO_FRAME, id=uuid4()),
        ],
    )
    def test_delete_media_annotations_not_found(
        self, media, fxt_get_project, fxt_media_service, fxt_dataset_service, fxt_client
    ):
        fxt_media_service.get_media_by_id.return_value = media
        fxt_dataset_service.delete_dataset_item_annotations.side_effect = ResourceNotFoundError(
            ResourceType.DATASET_ITEM, str(media.id)
        )

        response = fxt_client.delete(f"/api/projects/{str(uuid4())}/dataset/media/{str(media.id)}/annotations")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_media_service.get_media_by_id.assert_called_once_with(project_id=fxt_get_project.id, media_id=media.id)
        fxt_dataset_service.delete_dataset_item_annotations.assert_called_once_with(
            project=fxt_get_project,
            dataset_item_id=media.id,
        )

    @pytest.mark.parametrize(
        "media",
        [
            MagicMock(spec=Image, type=MediaType.IMAGE, id=uuid4()),
            MagicMock(spec=VideoFrame, type=MediaType.VIDEO_FRAME, id=uuid4()),
        ],
    )
    def test_list_video_frames_wrong_type(self, media, fxt_get_project, fxt_media_service, fxt_client):
        fxt_media_service.get_media_by_id.return_value = media

        response = fxt_client.get(f"/api/projects/{str(uuid4())}/dataset/media/{str(media.id)}/frames")

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        fxt_media_service.get_media_by_id.assert_called_once_with(project_id=fxt_get_project.id, media_id=media.id)
        fxt_media_service.list_annotated_video_frames_by_video_id.assert_not_called()

    def test_list_video_frames(self, fxt_get_project, fxt_media_service, fxt_client):
        video_frame_id = uuid4()
        label_id = uuid4()
        video = MagicMock(spec=Video, type=MediaType.VIDEO, id=uuid4())
        fxt_media_service.get_media_by_id.return_value = video

        dataset_item = MagicMock(
            spec=DatasetItem,
            user_reviewed=True,
            prediction_model_id=None,
            annotation_data=[
                DatasetItemAnnotation(
                    labels=[LabelReference(id=label_id)],
                    shape=Rectangle(type="rectangle", x=0, y=0, width=10, height=10),
                )
            ],
            subset=DatasetItemSubset.UNASSIGNED,
        )
        video_frame = MagicMock(spec=VideoFrame, type=MediaType.VIDEO_FRAME, id=video_frame_id, frame_index=5)

        fxt_media_service.list_annotated_video_frames_by_video_id.return_value = [(dataset_item, video_frame)]

        response = fxt_client.get(
            f"/api/projects/{str(uuid4())}/dataset/media/{str(video.id)}/frames?frame_index_from=1&frame_index_to=9"
        )

        assert response.status_code == status.HTTP_200_OK
        assert response.json() == [
            {
                "media_id": str(video_frame_id),
                "frame_index": 5,
                "annotation_data": {
                    "annotations": [
                        {
                            "confidences": None,
                            "labels": [{"id": str(label_id)}],
                            "shape": {"height": 10, "type": "rectangle", "width": 10, "x": 0, "y": 0},
                        }
                    ],
                    "prediction_model_id": None,
                    "user_reviewed": True,
                    "subset": "unassigned",
                },
            }
        ]
        fxt_media_service.get_media_by_id.assert_called_once_with(project_id=fxt_get_project.id, media_id=video.id)
        fxt_media_service.list_annotated_video_frames_by_video_id.assert_called_once_with(
            project=fxt_get_project,
            video_id=video.id,
            frame_index_from=1,
            frame_index_to=9,
        )

    @pytest.mark.parametrize(
        "predict_path_suffix",
        ["media:predict", "media/media:predict"],
        ids=["new_path", "deprecated_path"],
    )
    def test_media_predict(
        self,
        predict_path_suffix,
        fxt_get_project,
        fxt_media_prediction_service,
        fxt_inference_media_limit,
        fxt_client,
    ) -> None:
        label_id = uuid4()
        model_id = uuid4()
        media_id = uuid4()
        request = MediaListPredictionRequest(
            model_id=model_id,
            media=[MediaPredictionRequest(media_id=media_id, range=None)],
            device="AUTO",
        )

        fxt_inference_media_limit(10)

        fxt_media_prediction_service.predict_media.return_value = BatchInferenceResult(
            predictions=[
                BatchInferencePrediction(
                    media=BatchInferenceMedia(id=media_id),
                    prediction=[
                        DatasetItemAnnotation(
                            labels=[LabelReference(id=label_id)],
                            shape=Rectangle(type="rectangle", x=0, y=0, width=10, height=10),
                        )
                    ],
                )
            ]
        )

        response = fxt_client.post(
            f"/api/projects/{str(uuid4())}/dataset/{predict_path_suffix}",
            json=request.model_dump(mode="json"),
        )

        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {
            "predictions": [
                {
                    "media": {
                        "id": str(media_id),
                        "frame_index": None,
                    },
                    "prediction": [
                        {
                            "confidences": None,
                            "labels": [
                                {
                                    "id": str(label_id),
                                },
                            ],
                            "shape": {
                                "height": 10,
                                "type": "rectangle",
                                "width": 10,
                                "x": 0,
                                "y": 0,
                            },
                        },
                    ],
                },
            ]
        }

        fxt_media_prediction_service.predict_media.assert_called_once_with(
            project=fxt_get_project,
            request=request,
            device=DeviceInfo(type=DeviceType.AUTO, name="AUTO", memory=None, index=None),
        )

    def test_media_predict_video_range_error(
        self, fxt_get_project, fxt_media_prediction_service, fxt_inference_media_limit, fxt_client
    ) -> None:
        media_id = uuid4()
        request = MediaListPredictionRequest(
            model_id=uuid4(),
            media=[MediaPredictionRequest(media_id=media_id, range=None)],
            device="AUTO",
        )

        fxt_inference_media_limit(10)

        fxt_media_prediction_service.predict_media.side_effect = VideoRangeError(
            resource_id=str(media_id), message="Frame range can be specified only for videos."
        )
        response = fxt_client.post(
            f"/api/projects/{str(uuid4())}/dataset/media:predict",
            json=request.model_dump(mode="json"),
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert response.json() == {"detail": "Frame range can be specified only for videos."}

        fxt_media_prediction_service.predict_media.assert_called_once_with(
            project=fxt_get_project,
            request=request,
            device=DeviceInfo(type=DeviceType.AUTO, name="AUTO", memory=None, index=None),
        )

    def test_media_predict_limit_exceeded(
        self, fxt_get_project, fxt_media_prediction_service, fxt_inference_media_limit, fxt_client
    ) -> None:
        media_id = uuid4()
        request = MediaListPredictionRequest(
            model_id=uuid4(),
            media=[
                MediaPredictionRequest(media_id=media_id, range=VideoRange(start_frame=0, end_frame=100, stride=10))
            ],
            device="AUTO",
        )

        fxt_inference_media_limit(3)

        response = fxt_client.post(
            f"/api/projects/{str(uuid4())}/dataset/media:predict",
            json=request.model_dump(mode="json"),
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert response.json() == {
            "detail": "Too many media items to predict, requested number is 11 while limit is 3. "
            + "Please reduce the number of media or frame range size or set "
            + "INFERENCE_MEDIA_LIMIT environment variable with higher value ."
        }

        fxt_media_prediction_service.predict_media.assert_not_called()

    def test_media_predict_inference_busy(
        self, fxt_get_project, fxt_media_prediction_service, fxt_inference_media_limit, fxt_client
    ) -> None:
        request = MediaListPredictionRequest(
            model_id=uuid4(),
            media=[MediaPredictionRequest(media_id=uuid4(), range=None)],
            device="AUTO",
        )

        fxt_inference_media_limit(10)

        fxt_media_prediction_service.predict_media.side_effect = InferenceBusyError()

        response = fxt_client.post(
            f"/api/projects/{str(uuid4())}/dataset/media:predict",
            json=request.model_dump(mode="json"),
        )

        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert response.json() == {
            "detail": "Inference request timed out waiting for the model lock. Another inference is in "
            + "progress or model is not loaded yet."
        }

    def test_media_predict_with_confidence_threshold(
        self, fxt_get_project, fxt_media_prediction_service, fxt_inference_media_limit, fxt_client
    ) -> None:
        request = MediaListPredictionRequest(
            model_id=uuid4(),
            media=[MediaPredictionRequest(media_id=uuid4(), range=None)],
            device="AUTO",
            confidence_threshold=0.8,
        )

        fxt_inference_media_limit(10)
        fxt_media_prediction_service.predict_media.return_value = BatchInferenceResult(predictions=[])

        response = fxt_client.post(
            f"/api/projects/{str(uuid4())}/dataset/media:predict",
            json=request.model_dump(mode="json"),
        )

        assert response.status_code == status.HTTP_200_OK
        assert fxt_media_prediction_service.predict_media.call_args.kwargs["request"].confidence_threshold == 0.8

    @pytest.mark.parametrize("confidence_threshold", [-0.1, 1.5, "high"])
    def test_media_predict_invalid_confidence_threshold(
        self, confidence_threshold, fxt_get_project, fxt_media_prediction_service, fxt_inference_media_limit, fxt_client
    ) -> None:
        fxt_inference_media_limit(10)

        response = fxt_client.post(
            f"/api/projects/{str(uuid4())}/dataset/media:predict",
            json={
                "model_id": str(uuid4()),
                "media": [{"media_id": str(uuid4()), "range": None}],
                "device": "AUTO",
                "confidence_threshold": confidence_threshold,
            },
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
        fxt_media_prediction_service.predict_media.assert_not_called()

    def test_media_embeddings_image_success(
        self, fxt_image_media, fxt_get_project, fxt_media_service, fxt_media_segment_service, fxt_client
    ) -> None:
        fxt_media_service.get_media_by_id.return_value = fxt_image_media
        fxt_media_segment_service.encode_media.return_value = b"embeddings-bytes"

        response = fxt_client.get(f"/api/projects/{str(uuid4())}/dataset/media/{str(fxt_image_media.id)}/embeddings")

        assert response.status_code == status.HTTP_200_OK
        assert response.headers.get("content-type") == "application/octet-stream"
        assert f"{fxt_image_media.id}_embeddings.safetensors" in response.headers.get("content-disposition", "")
        assert response.content == b"embeddings-bytes"

        fxt_media_service.get_media_by_id.assert_called_once_with(
            project_id=fxt_get_project.id, media_id=fxt_image_media.id
        )
        fxt_media_segment_service.encode_media.assert_called_once_with(
            project=fxt_get_project, media=fxt_image_media, device=DeviceInfo.cpu()
        )

    def test_media_embeddings_video_frame_success(
        self, fxt_get_project, fxt_media_service, fxt_media_segment_service, fxt_client
    ) -> None:
        video_id = uuid4()
        media = MagicMock(spec=Video, id=video_id, format=VideoFormat.MP4, type=MediaType.VIDEO, frame_count=100)
        fxt_media_service.get_media_by_id.return_value = media
        # No annotated frame exists, so a NotAnnotatedVideoFrame is segmented on the fly
        fxt_media_service.get_video_frame_by_video_id_and_index.return_value = None
        fxt_media_segment_service.encode_media.return_value = b"frame-embeddings"

        response = fxt_client.get(
            f"/api/projects/{str(uuid4())}/dataset/media/{str(video_id)}/embeddings?frame_index=10"
        )

        assert response.status_code == status.HTTP_200_OK
        assert response.headers.get("content-type") == "application/octet-stream"
        assert f"{video_id}_10_embeddings.safetensors" in response.headers.get("content-disposition", "")
        assert response.content == b"frame-embeddings"

        fxt_media_service.get_media_by_id.assert_called_once_with(project_id=fxt_get_project.id, media_id=video_id)
        fxt_media_service.get_video_frame_by_video_id_and_index.assert_called_once_with(
            project=fxt_get_project, video_id=video_id, frame_index=10
        )
        fxt_media_segment_service.encode_media.assert_called_once_with(
            project=fxt_get_project, media=ANY, device=DeviceInfo.cpu()
        )

    def test_media_embeddings_video_without_frame_index(
        self, fxt_video_media, fxt_get_project, fxt_media_service, fxt_media_segment_service, fxt_client
    ) -> None:
        fxt_media_service.get_media_by_id.return_value = fxt_video_media

        response = fxt_client.get(f"/api/projects/{str(uuid4())}/dataset/media/{str(fxt_video_media.id)}/embeddings")

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert response.json() == {"detail": "Video frame index is not provided."}
        fxt_media_segment_service.encode_media.assert_not_called()

    def test_media_embeddings_media_not_found(
        self, fxt_get_project, fxt_media_service, fxt_media_segment_service, fxt_client
    ) -> None:
        media_id = uuid4()
        fxt_media_service.get_media_by_id.side_effect = ResourceNotFoundError(ResourceType.MEDIA, str(media_id))

        response = fxt_client.get(f"/api/projects/{str(uuid4())}/dataset/media/{str(media_id)}/embeddings")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_media_service.get_media_by_id.assert_called_once_with(project_id=fxt_get_project.id, media_id=media_id)
        fxt_media_segment_service.encode_media.assert_not_called()

    def test_media_embeddings_video_range_error(
        self, fxt_image_media, fxt_get_project, fxt_media_service, fxt_media_segment_service, fxt_client
    ) -> None:
        fxt_media_service.get_media_by_id.return_value = fxt_image_media
        fxt_media_segment_service.encode_media.side_effect = VideoRangeError(
            resource_id=str(fxt_image_media.id), message="Frame range can be specified only for videos."
        )

        response = fxt_client.get(f"/api/projects/{str(uuid4())}/dataset/media/{str(fxt_image_media.id)}/embeddings")

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert response.json() == {"detail": "Frame range can be specified only for videos."}

    def test_media_embeddings_binary_not_found(
        self, fxt_image_media, fxt_get_project, fxt_media_service, fxt_media_segment_service, fxt_client
    ) -> None:
        fxt_media_service.get_media_by_id.return_value = fxt_image_media
        fxt_media_segment_service.encode_media.side_effect = BinaryNotFoundError(
            f"Media {str(fxt_image_media.id)} binary cannot be found"
        )

        response = fxt_client.get(f"/api/projects/{str(uuid4())}/dataset/media/{str(fxt_image_media.id)}/embeddings")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert response.json() == {"detail": f"Media {str(fxt_image_media.id)} binary cannot be found"}
