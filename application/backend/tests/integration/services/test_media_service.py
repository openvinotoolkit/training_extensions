# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os.path
import tempfile
from collections.abc import Callable
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import cv2
import numpy as np
import pytest
from PIL import ExifTags
from PIL import Image as PILImage
from sqlalchemy.orm import Session

from app.db.schema import DatasetItemDB, DatasetItemLabelDB, MediaDB, PipelineDB
from app.models import DatasetItemAnnotationStatus, DatasetItemSubset, Pipeline, Project, Video
from app.models.media import ImageFormat, MediaSortBy, MediaType, SortDirection, VideoFormat
from app.services.base import ResourceNotFoundError, ResourceType
from app.services.media_service import ImageMetadata, InvalidImageError, MediaFilters, MediaService


@pytest.fixture
def fxt_project_with_pipeline(
    fxt_db_projects,
    fxt_db_labels,
    fxt_project_service,
    fxt_pipeline_service,
    fxt_db_sources,
    fxt_db_sinks,
    fxt_db_models,
    db_session,
) -> tuple[Project, Pipeline]:
    """Fixture to create a Project."""

    db_project = fxt_db_projects[0]
    db_session.add(db_project)
    db_session.flush()

    db_model = fxt_db_models[0]
    db_model.project_id = db_project.id
    for label in fxt_db_labels:
        label.project_id = db_project.id
    db_session.add_all([db_model, *fxt_db_labels])
    db_session.flush()

    db_pipeline = PipelineDB(project_id=db_project.id)
    db_pipeline.source = fxt_db_sources[0]
    db_pipeline.sink = fxt_db_sinks[0]
    db_pipeline.model_revision = db_model
    db_session.add(db_pipeline)
    db_session.flush()

    return (
        fxt_project_service.get_project_by_id(UUID(db_project.id)),
        fxt_pipeline_service.get_pipeline_by_id(UUID(db_project.id)),
    )


@pytest.fixture
def fxt_media_factory(db_session) -> Callable[[str, list[dict]], list[MediaDB]]:
    """Returns a callable that creates and persists MediaDB objects for a project."""

    def _create_media(project_id: str, configs: list[dict]) -> list[MediaDB]:
        items = []
        for config in configs:
            m = MediaDB(**config)
            m.project_id = project_id
            m.created_at = datetime.fromisoformat("2025-02-01T00:00:00Z")
            items.append(m)
        db_session.add_all(items)
        db_session.flush()
        return items

    return _create_media


@pytest.fixture
def fxt_project_with_media(fxt_project_with_pipeline, fxt_media_factory, db_session) -> tuple[Project, list[MediaDB]]:
    project, _ = fxt_project_with_pipeline

    configs = [
        {"type": "image", "name": "test1", "format": "jpg", "size": 1024, "width": 1024, "height": 768},
        {"type": "image", "name": "test2", "format": "jpg", "size": 1024, "width": 1024, "height": 768},
        {"type": "image", "name": "test3", "format": "jpg", "size": 1024, "width": 1024, "height": 768},
    ]
    video_config = {
        "id": str(uuid4()),
        "type": "video",
        "name": "test4",
        "format": "avi",
        "size": 1024,
        "width": 1024,
        "height": 768,
        "fps": 25.0,
        "frame_count": 100,
    }
    video_frame_config = {
        "video_id": video_config["id"],
        "type": "video_frame",
        "name": "test4_10",
        "format": "jpg",
        "size": 1024,
        "width": 1024,
        "height": 768,
        "frame_index": 20,
    }

    db_media_list = fxt_media_factory(
        str(project.id),
        [
            *configs,
            video_config,
            video_frame_config,
        ],
    )

    return project, db_media_list


@pytest.fixture
def fxt_video_frame(
    fxt_project_with_media: tuple[Project, list[MediaDB]], db_session
) -> Callable[[int], tuple[DatasetItemDB, MediaDB]]:
    def _create_video_frame(frame_index: int) -> tuple[DatasetItemDB, MediaDB]:
        project, db_media_list = fxt_project_with_media
        media = db_media_list[3]

        db_media = MediaDB(
            type="video_frame",
            name=f"test4_frame_{frame_index}",
            format="jpg",
            size=1024,
            width=1024,
            height=768,
            video_id=media.id,
            frame_index=frame_index,
        )
        db_media.project_id = str(project.id)
        db_media.created_at = datetime.fromisoformat("2025-02-01T00:00:00Z")

        db_session.add(db_media)
        db_session.flush()

        db_dataset_item = DatasetItemDB(id=db_media.id, project_id=str(project.id), subset="unassigned")

        db_session.add(db_dataset_item)
        db_session.flush()

        return db_dataset_item, db_media

    return _create_video_frame


@pytest.fixture
def fxt_project_with_annotation_status_items(
    fxt_project_with_pipeline, db_session
) -> tuple[Project, list[DatasetItemDB]]:
    """Fixture with dataset items covering all annotation statuses."""
    project, _ = fxt_project_with_pipeline

    # Unannotated items (annotation_data is null) - don't set annotation_data at all
    unannotated_items = [
        DatasetItemDB(
            subset="unassigned",
            user_reviewed=False,
            project_id=str(project.id),
            created_at=datetime.fromisoformat("2025-02-01T00:00:00Z"),
        ),
        DatasetItemDB(
            subset="unassigned",
            user_reviewed=False,
            project_id=str(project.id),
            created_at=datetime.fromisoformat("2025-02-01T00:00:00Z"),
        ),
    ]

    # Reviewed items (annotation_data is not null and user_reviewed is True)
    reviewed_items = [
        DatasetItemDB(
            subset="unassigned",
            annotation_data=[{"labels": [{"id": str(project.task.labels[0].id)}], "shape": {"type": "full_image"}}],
            user_reviewed=True,
            project_id=str(project.id),
            created_at=datetime.fromisoformat("2025-02-01T00:00:00Z"),
        ),
        DatasetItemDB(
            subset="unassigned",
            annotation_data=[{"labels": [{"id": str(project.task.labels[0].id)}], "shape": {"type": "full_image"}}],
            user_reviewed=True,
            project_id=str(project.id),
            created_at=datetime.fromisoformat("2025-02-01T00:00:00Z"),
        ),
        DatasetItemDB(
            subset="unassigned",
            annotation_data=[{"labels": [{"id": str(project.task.labels[0].id)}], "shape": {"type": "full_image"}}],
            user_reviewed=True,
            project_id=str(project.id),
            created_at=datetime.fromisoformat("2025-02-01T00:00:00Z"),
        ),
    ]

    db_dataset_items = []
    for list, name in zip([unannotated_items, reviewed_items], ["unannotated", "reviewed"]):
        for idx, dataset_item in enumerate(list):
            db_media = MediaDB(
                type="image",
                name=f"{name}{idx + 1}",
                format="jpg",
                size=1024,
                width=1024,
                height=768,
                project_id=str(project.id),
                created_at=datetime.fromisoformat("2025-02-01T00:00:00Z"),
            )
            db_session.add(db_media)
            db_session.flush()

            dataset_item.id = db_media.id
            db_dataset_items.append(dataset_item)

    db_session.add_all(db_dataset_items)
    db_session.flush()

    # Link labels to annotated dataset items
    for item in [*reviewed_items]:
        db_session.add(DatasetItemLabelDB(dataset_item_id=item.id, label_id=str(project.task.labels[0].id)))
    db_session.flush()

    return project, db_dataset_items


@pytest.fixture
def fxt_project_with_labeled_dataset_items(
    fxt_project_with_pipeline, db_session
) -> tuple[Project, list[DatasetItemDB]]:
    """Fixture to create a project with multiple labeled dataset items for testing label filtering."""
    project, _ = fxt_project_with_pipeline

    # Ensure we have at least 2 labels
    assert len(project.task.labels) >= 2, "Project must have at least 2 labels for this fixture"

    label_0_id = str(project.task.labels[0].id)
    label_1_id = str(project.task.labels[1].id)

    configs: list[tuple[dict[str, Any], dict[str, Any]]] = [
        # Item 0: No annotations
        (
            {"type": "image", "name": "item_no_labels", "format": "jpg", "size": 1024, "width": 1024, "height": 768},
            {"subset": "unassigned"},
        ),
        # Item 1: Has label_0
        (
            {
                "type": "image",
                "name": "item_label_0",
                "format": "jpg",
                "size": 1024,
                "width": 1024,
                "height": 768,
            },
            {
                "subset": "unassigned",
                "annotation_data": [{"labels": [{"id": label_0_id}], "shape": {"type": "full_image"}}],
            },
        ),
        # Item 2: Has label_1
        (
            {
                "type": "image",
                "name": "item_label_1",
                "format": "jpg",
                "size": 1024,
                "width": 1024,
                "height": 768,
            },
            {
                "subset": "unassigned",
                "annotation_data": [{"labels": [{"id": label_1_id}], "shape": {"type": "full_image"}}],
            },
        ),
        # Item 3: Has both label_0 and label_1
        (
            {
                "type": "image",
                "name": "item_both_labels",
                "format": "jpg",
                "size": 1024,
                "width": 1024,
                "height": 768,
            },
            {
                "subset": "unassigned",
                "annotation_data": [
                    {
                        "labels": [{"id": label_0_id}],
                        "shape": {"type": "rectangle", "x": 0, "y": 0, "width": 10, "height": 10},
                    },
                    {
                        "labels": [{"id": label_1_id}],
                        "shape": {"type": "rectangle", "x": 20, "y": 20, "width": 10, "height": 10},
                    },
                ],
            },
        ),
    ]

    db_dataset_items = []
    for index, (media_config, dataset_item_config) in enumerate(configs):
        media = MediaDB(**media_config)
        media.project_id = str(project.id)
        db_session.add(media)
        db_session.flush()

        dataset_item = DatasetItemDB(**dataset_item_config)
        dataset_item.id = str(media.id)
        dataset_item.project_id = str(project.id)
        dataset_item.created_at = datetime.fromisoformat("2025-02-01T00:00:00Z")
        db_dataset_items.append(dataset_item)
    db_session.add_all(db_dataset_items)
    db_session.flush()

    # Link labels to dataset items
    db_session.add(DatasetItemLabelDB(dataset_item_id=db_dataset_items[1].id, label_id=label_0_id))
    db_session.add(DatasetItemLabelDB(dataset_item_id=db_dataset_items[2].id, label_id=label_1_id))
    db_session.add(DatasetItemLabelDB(dataset_item_id=db_dataset_items[3].id, label_id=label_0_id))
    db_session.add(DatasetItemLabelDB(dataset_item_id=db_dataset_items[3].id, label_id=label_1_id))
    db_session.flush()

    return project, db_dataset_items


@pytest.fixture
def fxt_project_with_labeled_video(fxt_project_with_pipeline, db_session) -> tuple[Project, MediaDB, list[MediaDB]]:
    """Fixture with a video whose frames carry labels, plus a standalone labeled image.

    Layout:
      - A video (no dataset item of its own).
      - Two frames of that video: frame 10 annotated with label_0, frame 20 annotated with label_0.
      - A standalone image annotated with label_1.

    Returns the project, the video MediaDB and the list of frame MediaDB objects.
    """
    project, _ = fxt_project_with_pipeline
    assert len(project.task.labels) >= 2, "Project must have at least 2 labels for this fixture"

    label_0_id = str(project.task.labels[0].id)
    label_1_id = str(project.task.labels[1].id)

    created_at = datetime.fromisoformat("2025-02-01T00:00:00Z")

    # Video (no dataset item of its own)
    video = MediaDB(
        type="video",
        name="labeled_video",
        format="avi",
        size=1024,
        width=1024,
        height=768,
        fps=25.0,
        frame_count=100,
        project_id=str(project.id),
        created_at=created_at,
    )
    db_session.add(video)
    db_session.flush()

    # Two frames of the video, both annotated with label_0
    frames: list[MediaDB] = []
    for frame_index in (10, 20):
        frame = MediaDB(
            type="video_frame",
            name=f"labeled_video_frame_{frame_index}",
            format="jpg",
            size=1024,
            width=1024,
            height=768,
            video_id=video.id,
            frame_index=frame_index,
            project_id=str(project.id),
            created_at=created_at,
        )
        db_session.add(frame)
        db_session.flush()
        frame_item = DatasetItemDB(
            id=frame.id,
            project_id=str(project.id),
            subset="unassigned",
            annotation_data=[{"labels": [{"id": label_0_id}], "shape": {"type": "full_image"}}],
            user_reviewed=True,
            created_at=created_at,
        )
        db_session.add(frame_item)
        db_session.flush()
        db_session.add(DatasetItemLabelDB(dataset_item_id=frame.id, label_id=label_0_id))
        frames.append(frame)

    # Standalone image annotated with label_1
    image = MediaDB(
        type="image",
        name="labeled_image",
        format="jpg",
        size=1024,
        width=1024,
        height=768,
        project_id=str(project.id),
        created_at=created_at,
    )
    db_session.add(image)
    db_session.flush()
    image_item = DatasetItemDB(
        id=image.id,
        project_id=str(project.id),
        subset="unassigned",
        annotation_data=[{"labels": [{"id": label_1_id}], "shape": {"type": "full_image"}}],
        user_reviewed=True,
        created_at=created_at,
    )
    db_session.add(image_item)
    db_session.flush()
    db_session.add(DatasetItemLabelDB(dataset_item_id=image.id, label_id=label_1_id))
    db_session.flush()

    return project, video, frames


@pytest.fixture
def fxt_project_with_subset_items(fxt_project_with_pipeline, db_session) -> tuple[Project, list[DatasetItemDB]]:
    """Fixture with dataset items covering all subset types."""
    project, _ = fxt_project_with_pipeline

    # Unassigned items
    unassigned_items = [
        DatasetItemDB(
            subset=DatasetItemSubset.UNASSIGNED,
            user_reviewed=False,
            project_id=str(project.id),
            created_at=datetime.fromisoformat("2025-02-01T00:00:00Z"),
        ),
        DatasetItemDB(
            subset=DatasetItemSubset.UNASSIGNED,
            user_reviewed=False,
            project_id=str(project.id),
            created_at=datetime.fromisoformat("2025-02-02T00:00:00Z"),
        ),
    ]

    # Training items
    training_items = [
        DatasetItemDB(
            subset=DatasetItemSubset.TRAINING,
            user_reviewed=False,
            project_id=str(project.id),
            created_at=datetime.fromisoformat("2025-02-03T00:00:00Z"),
        ),
        DatasetItemDB(
            subset=DatasetItemSubset.TRAINING,
            user_reviewed=False,
            project_id=str(project.id),
            created_at=datetime.fromisoformat("2025-02-04T00:00:00Z"),
        ),
        DatasetItemDB(
            subset=DatasetItemSubset.TRAINING,
            user_reviewed=False,
            project_id=str(project.id),
            created_at=datetime.fromisoformat("2025-02-05T00:00:00Z"),
        ),
    ]

    # Validation items
    validation_items = [
        DatasetItemDB(
            subset=DatasetItemSubset.VALIDATION,
            user_reviewed=False,
            project_id=str(project.id),
            created_at=datetime.fromisoformat("2025-02-06T00:00:00Z"),
        ),
        DatasetItemDB(
            subset=DatasetItemSubset.VALIDATION,
            user_reviewed=False,
            project_id=str(project.id),
            created_at=datetime.fromisoformat("2025-02-07T00:00:00Z"),
        ),
    ]

    # Testing items
    testing_items = [
        DatasetItemDB(
            subset=DatasetItemSubset.TESTING,
            user_reviewed=False,
            project_id=str(project.id),
            created_at=datetime.fromisoformat("2025-02-08T00:00:00Z"),
        ),
    ]
    db_dataset_items = []
    for list in [unassigned_items, training_items, validation_items, testing_items]:
        for idx, dataset_item in enumerate(list):
            db_media = MediaDB(
                type="image",
                name=f"{dataset_item.subset}{idx + 1}",
                format="jpg",
                size=1024,
                width=1024,
                height=768,
                project_id=str(project.id),
                created_at=datetime.fromisoformat("2025-02-08T00:00:00Z"),
            )
            db_session.add(db_media)
            db_session.flush()
            dataset_item.id = db_media.id
            db_dataset_items.append(dataset_item)

    db_session.add_all(db_dataset_items)
    db_session.flush()

    return project, db_dataset_items


@pytest.fixture
def fxt_project_with_media_at_distinct_dates(fxt_project_with_pipeline, db_session) -> tuple[Project, list[MediaDB]]:
    """Fixture with three images uploaded on distinct, known dates (test1 oldest, test3 newest)."""
    project, _ = fxt_project_with_pipeline

    db_media_list = [
        MediaDB(
            type="image",
            name=name,
            format="jpg",
            size=1024,
            width=1024,
            height=768,
            project_id=str(project.id),
        )
        for name in ("test1", "test2", "test3")
    ]
    db_session.add_all(db_media_list)
    db_session.flush()
    for db_media, created_at in zip(
        db_media_list,
        [
            datetime.fromisoformat("2025-02-01T00:00:00Z"),
            datetime.fromisoformat("2025-02-02T00:00:00Z"),
            datetime.fromisoformat("2025-02-03T00:00:00Z"),
        ],
    ):
        db_media.created_at = created_at
    db_session.add_all(db_media_list)
    db_session.flush()

    return project, db_media_list


@pytest.fixture
def fxt_video_data() -> Callable[[Path], None]:
    def _generate_video_file(path: Path) -> None:
        duration_sec = 4
        fps = 25
        width = 640
        height = 480

        fourccc = cv2.VideoWriter.fourcc(*"MJPG")
        writer = cv2.VideoWriter(str(path), fourccc, fps, (width, height), isColor=True)
        assert writer.isOpened()

        black_frame = np.zeros((height, width, 3), dtype=np.uint8)
        for _ in range(duration_sec * fps):
            writer.write(black_frame)
        writer.release()

    return _generate_video_file


class TestMediaServiceIntegration:
    """Integration tests for MediaService."""

    @pytest.mark.parametrize("use_pipeline_source", [True, False])
    @pytest.mark.parametrize("format", ImageFormat)
    def test_create_image(
        self,
        tmp_path: Path,
        fxt_media_service: MediaService,
        fxt_project_with_pipeline: tuple[Project, Pipeline],
        db_session: Session,
        format: ImageFormat,
        use_pipeline_source: bool,
    ) -> None:
        """Test creating a media."""
        rng = np.random.default_rng(seed=42)
        image_data = rng.integers(0, 255, (64, 96, 3), dtype=np.uint8)
        image = PILImage.fromarray(image_data, mode="RGB")
        image.getexif()[ExifTags.Base.Software] = "Intel Geti"

        original_file_path = tmp_path / f"original.{format}"
        image.save(original_file_path, exif=image.getexif())
        original_bytes = original_file_path.read_bytes()

        project, pipeline = fxt_project_with_pipeline

        created_media = fxt_media_service.create_image(
            ImageMetadata(
                project_id=project.id,
                name="test",
                image_format=format,
                data=BytesIO(original_bytes),
                source_id=pipeline.source_id if use_pipeline_source else None,
            )
        )

        binary_file_path = tmp_path / f"projects/{project.id}/dataset/{created_media.id}.{format}"
        assert binary_file_path.exists()
        assert binary_file_path.read_bytes() == original_bytes

        media = db_session.get(MediaDB, str(created_media.id))
        assert media is not None
        assert (
            media.id == str(created_media.id)
            and media.project_id == str(project.id)
            and media.type == "image"
            and media.name == "test"
            and media.format == format
            and media.width == 96
            and media.height == 64
        )
        if use_pipeline_source:
            assert media.source_id == str(pipeline.source_id)
        else:
            assert media.source_id is None

        binary_file_path = tmp_path / f"projects/{project.id}/dataset/{created_media.id}.{format}"
        assert os.path.exists(binary_file_path)
        assert created_media.size == os.path.getsize(binary_file_path)

        with PILImage.open(binary_file_path) as stored_image:
            stored_exif = stored_image.getexif()
        assert stored_exif is not None
        if format != ImageFormat.BMP:  # BMP images do not support EXIF information
            assert stored_exif[ExifTags.Base.Software] == "Intel Geti"

        thumbnail_file_path = tmp_path / f"projects/{project.id}/dataset/{created_media.id}-thumb.jpg"
        assert os.path.exists(thumbnail_file_path)

    def test_create_image_16bit_png_thumbnail(
        self,
        tmp_path: Path,
        fxt_media_service: MediaService,
        fxt_project_with_pipeline: tuple[Project, Pipeline],
    ) -> None:
        """Test that a thumbnail is generated correctly for a 16-bit PNG image (mode I;16).

        Specifically, the full dynamic range of the 16-bit source must be
        preserved through normalization — a naive PIL convert("RGB") only
        keeps the most-significant byte, producing washed-out thumbnails.
        """
        # Create a 16-bit grayscale image that uses the full 0-65535 range
        rng = np.random.default_rng(seed=0)
        data_16bit = rng.integers(0, 65535, (512, 512), dtype=np.uint16)
        image = PILImage.fromarray(data_16bit, mode="I;16")

        # Ensure the image is recognised as 16-bit by PIL
        assert image.mode == "I;16"

        project, _ = fxt_project_with_pipeline

        created_media = fxt_media_service.create_image(
            ImageMetadata(
                project_id=project.id,
                name="test_16bit",
                image_format=ImageFormat.PNG,
                data=image,
            )
        )

        # The thumbnail must exist and be a valid JPEG readable by PIL
        thumbnail_file_path = tmp_path / f"projects/{project.id}/dataset/{created_media.id}-thumb.jpg"
        assert os.path.exists(thumbnail_file_path), "Thumbnail file was not created for 16-bit PNG"

        with PILImage.open(thumbnail_file_path) as thumb:
            assert thumb.format == "JPEG"
            assert thumb.mode == "RGB"

            # The normalized thumbnail must use a wide tonal range.
            # A naive MSB-only conversion collapses most pixels to black (max≈255,
            # but the vast majority of values cluster near 0), while correct
            # normalization spreads values across the full 0-255 range.
            arr = np.array(thumb)
            assert arr.min() < 10, "Shadow end of range missing — normalization likely clipped low values"
            assert arr.max() > 245, "Highlight end of range missing — normalization likely clipped high values"

    def test_create_image_from_uint16_3channel_ndarray(
        self,
        tmp_path: Path,
        fxt_media_service: MediaService,
        fxt_project_with_pipeline: tuple[Project, Pipeline],
        db_session: Session,
    ) -> None:
        """16-bit 3-channel ndarrays from datumaro are saved as single-channel 16-bit PNG.

        Datumaro returns grayscale 16-bit images as (H, W, 3) uint16 arrays with identical channels. Pillow rejects
        Image.fromarray on such arrays, so _read_image_from_ndarray must collapse them to (H, W) before conversion.
        The stored file must be:
          - readable by PIL
          - in a single-channel 16-bit mode (I;16)
          - sized correctly
          - accompanied by a valid JPEG thumbnail
        """
        rng = np.random.default_rng(seed=42)
        channel = rng.integers(0, 65535, (128, 96), dtype=np.uint16)
        data = np.stack([channel, channel, channel], axis=-1)  # (128, 96, 3) uint16

        project, _ = fxt_project_with_pipeline

        created_media = fxt_media_service.create_image(
            ImageMetadata(
                project_id=project.id,
                name="test_16bit_ndarray",
                image_format=ImageFormat.PNG,
                data=data,
            )
        )

        # DB record is correct
        db_media = db_session.get(MediaDB, str(created_media.id))
        assert db_media is not None
        assert db_media.width == 96
        assert db_media.height == 128
        assert db_media.format == "png"

        # Stored file is a valid single-channel 16-bit PNG
        binary_path = tmp_path / f"projects/{project.id}/dataset/{created_media.id}.png"
        assert binary_path.exists()
        with PILImage.open(binary_path) as img:
            assert img.mode in ("I;16", "I;16B", "I;16L", "I"), f"Expected single-channel 16-bit mode, got {img.mode}"
            assert img.width == 96
            assert img.height == 128
            # Pixel values must match the original channel (no bit-depth truncation)
            np.testing.assert_array_equal(np.array(img), channel)

        # Thumbnail exists and is a valid JPEG
        thumb_path = tmp_path / f"projects/{project.id}/dataset/{created_media.id}-thumb.jpg"
        assert thumb_path.exists()
        with PILImage.open(thumb_path) as thumb:
            assert thumb.format == "JPEG"
            assert thumb.mode == "RGB"

    @pytest.mark.parametrize("format", [ImageFormat.TIF, ImageFormat.TIFF])
    def test_create_tiff_image_with_problematic_exif(
        self,
        tmp_path: Path,
        fxt_media_service: MediaService,
        fxt_project_with_pipeline: tuple[Project, Pipeline],
        db_session: Session,
        format: ImageFormat,
    ) -> None:
        """Test that TIFF images with EXIF data that cannot be re-serialized by
        PIL's libtiff encoder are still saved successfully (without EXIF).

        Regression test for RuntimeError: 'Error setting from dictionary' when
        uploading certain TIFF files whose IFD contains tags incompatible with
        the libtiff encoder round-trip.
        """
        image = PILImage.new("RGB", (896, 768))

        # Inject a problematic EXIF tag that triggers the libtiff encoder error.
        # Tag 0x8769 (ExifOffset / ExifIFD) with an invalid dict value causes
        # PIL's libtiff encoder to raise RuntimeError when saving.
        exif = image.getexif()
        exif[ExifTags.Base.Software] = "Intel Geti"
        exif.get_ifd(ExifTags.IFD.Exif)[ExifTags.Base.ExifVersion] = b"0232"
        image.info["exif"] = exif.tobytes()

        # Patch image.save to simulate the libtiff RuntimeError when exif= is
        # passed, then succeed on the retry without exif.
        original_save = PILImage.Image.save
        exif_save_attempted = False
        fallback_save_done = False

        def patched_save(self_img, *args, **kwargs):
            nonlocal exif_save_attempted, fallback_save_done
            if "exif" in kwargs:
                exif_save_attempted = True
                raise RuntimeError("Error setting from dictionary")
            # Track that the immediate retry (same path, no exif) succeeded
            if exif_save_attempted and not fallback_save_done:
                fallback_save_done = True
            return original_save(self_img, *args, **kwargs)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(PILImage.Image, "save", patched_save)

            project, _ = fxt_project_with_pipeline
            created_media = fxt_media_service.create_image(
                ImageMetadata(
                    project_id=project.id,
                    name="test_tiff_exif",
                    image_format=format,
                    data=image,
                )
            )

        assert exif_save_attempted, "Expected save with exif= to be attempted"
        assert fallback_save_done, "Expected fallback save without exif to succeed"

        media = db_session.get(MediaDB, str(created_media.id))
        assert media is not None
        assert media.width == 896
        assert media.height == 768
        assert media.format == format

        binary_file_path = tmp_path / f"projects/{project.id}/dataset/{created_media.id}.{format}"
        assert os.path.exists(binary_file_path)

    @pytest.mark.parametrize("use_pipeline_source", [True, False])
    @pytest.mark.parametrize("format", VideoFormat)
    def test_create_video(
        self,
        tmp_path: Path,
        fxt_projects_dir: Path,
        fxt_video_data: Callable[[Path], None],
        fxt_media_service: MediaService,
        fxt_project_with_pipeline: tuple[Project, Pipeline],
        db_session: Session,
        format: VideoFormat,
        use_pipeline_source: bool,
    ) -> None:
        """Test creating a video."""
        project, pipeline = fxt_project_with_pipeline

        # Generate video
        with tempfile.TemporaryDirectory(delete=True) as tmp_dir:
            tmp_file = Path(tmp_dir) / "video.avi"

            fxt_video_data(Path(tmp_file))
            with open(tmp_file, mode="rb") as data:
                created_media = fxt_media_service.create_video(
                    project_id=project.id,
                    name="test",
                    video_format=format,
                    data=data,
                    source_id=pipeline.source_id if use_pipeline_source else None,
                )

        media = db_session.get(MediaDB, str(created_media.id))
        assert media is not None
        assert (
            media.id == str(created_media.id)
            and media.project_id == str(project.id)
            and media.type == "video"
            and media.name == "test"
            and media.format == format
            and media.width == 640
            and media.height == 480
        )
        if use_pipeline_source:
            assert media.source_id == str(pipeline.source_id)
        else:
            assert media.source_id is None

        binary_file_path = tmp_path / f"projects/{project.id}/dataset/{created_media.id}.{format}"
        assert os.path.exists(binary_file_path)
        assert created_media.size == os.path.getsize(binary_file_path)

        # Generate thumbnail on video upload
        thumbnail_file_path = tmp_path / f"projects/{project.id}/dataset/{created_media.id}-thumb.jpg"
        assert os.path.exists(thumbnail_file_path)

    def test_create_media_invalid_image(
        self,
        fxt_media_service: MediaService,
        fxt_project_with_pipeline: tuple[Project, Pipeline],
        db_session: Session,
    ) -> None:
        """Test creating a media with invalid image."""
        project, _ = fxt_project_with_pipeline

        with pytest.raises(InvalidImageError):
            fxt_media_service.create_image(
                ImageMetadata(
                    project_id=project.id,
                    name="test",
                    image_format=ImageFormat.JPG,
                    data=BytesIO(b"123"),
                )
            )

    @pytest.mark.parametrize(
        "start_date, start_date_out_of_range",
        [
            (None, False),
            (datetime.fromisoformat("2025-01-01T00:00:00Z"), False),
            (datetime.fromisoformat("2025-02-02T00:00:00Z"), True),
        ],
    )
    @pytest.mark.parametrize(
        "end_date, end_date_out_of_range",
        [
            (None, False),
            (datetime.fromisoformat("2025-02-02T00:00:00Z"), False),
            (datetime.fromisoformat("2025-01-01T00:00:00Z"), True),
        ],
    )
    def test_count_media(
        self,
        fxt_media_service: MediaService,
        fxt_project_with_media: tuple[Project, list[MediaDB]],
        db_session: Session,
        start_date,
        start_date_out_of_range,
        end_date,
        end_date_out_of_range,
    ) -> None:
        """Test counting media."""
        project, db_media_list = fxt_project_with_media

        count = fxt_media_service.count_media(project=project, start_date=start_date, end_date=end_date)

        assert count == 0 if start_date_out_of_range or end_date_out_of_range else count == len(db_media_list)

    @pytest.mark.parametrize(
        "exclude_types, count", [([MediaType.IMAGE], 2), ([MediaType.VIDEO], 4), ([MediaType.VIDEO_FRAME], 4)]
    )
    def test_count_media_excluding(
        self,
        exclude_types: list[MediaType],
        count: int,
        fxt_media_service: MediaService,
        fxt_project_with_media: tuple[Project, list[MediaDB]],
        db_session: Session,
    ) -> None:
        """Test counting media with media types exclusion."""
        project, db_media_list = fxt_project_with_media

        result = fxt_media_service.count_media(project=project, exclude_types=exclude_types)

        assert result == count

    @pytest.mark.parametrize("limit, limit_out_of_range", [(10, False), (0, True)])
    @pytest.mark.parametrize("offset, offset_out_of_range", [(0, False), (10, True)])
    @pytest.mark.parametrize(
        "start_date, start_date_out_of_range",
        [
            (None, False),
            (datetime.fromisoformat("2025-01-01T00:00:00Z"), False),
            (datetime.fromisoformat("2025-02-02T00:00:00Z"), True),
        ],
    )
    @pytest.mark.parametrize(
        "end_date, end_date_out_of_range",
        [
            (None, False),
            (datetime.fromisoformat("2025-02-02T00:00:00Z"), False),
            (datetime.fromisoformat("2025-01-01T00:00:00Z"), True),
        ],
    )
    def test_list_media(
        self,
        fxt_media_service: MediaService,
        fxt_project_with_media: tuple[Project, list[MediaDB]],
        limit,
        limit_out_of_range,
        offset,
        offset_out_of_range,
        start_date,
        start_date_out_of_range,
        end_date,
        end_date_out_of_range,
    ) -> None:
        """Test listing media."""
        project, db_media_list = fxt_project_with_media

        media_list = fxt_media_service.list_media(
            project_id=project.id,
            filters=MediaFilters(
                limit=limit,
                offset=offset,
                start_date=start_date,
                end_date=end_date,
            ),
        )

        assert (
            len(media_list) == 0
            if start_date_out_of_range or end_date_out_of_range or limit_out_of_range or offset_out_of_range
            else len(media_list) == len(db_media_list)
        )

    def test_list_media_ids(
        self,
        fxt_media_service: MediaService,
        fxt_project_with_media: tuple[Project, list[MediaDB]],
    ) -> None:
        """list_media_ids should return the same (id, type) pairs as list_media, without loading full media rows."""
        project, _ = fxt_project_with_media

        media_list = fxt_media_service.list_media(project_id=project.id, filters=MediaFilters(limit=100))
        media_id_list = fxt_media_service.list_media_ids(project_id=project.id)

        assert set(media_id_list) == {(media.id, media.type) for media in media_list}

    @pytest.mark.parametrize(
        "annotation_status",
        [None, DatasetItemAnnotationStatus.WITH_ANNOTATIONS, DatasetItemAnnotationStatus.MISSING_ANNOTATIONS],
    )
    def test_list_media_with_annotated_frame_annotation_status(
        self,
        annotation_status: DatasetItemAnnotationStatus | None,
        fxt_media_service: MediaService,
        fxt_project_with_media: tuple[Project, list[MediaDB]],
        db_session: Session,
    ) -> None:
        """Test listing media should include the video that have annotated frame with each filter option."""
        project, db_media_list = fxt_project_with_media

        db_video_frame = next(db_media for db_media in db_media_list if db_media.type == MediaType.VIDEO_FRAME)
        db_dataset_item = DatasetItemDB(
            id=db_video_frame.id,
            project_id=str(project.id),
            subset="unassigned",
            annotation_data=[{"labels": [{"id": str(uuid4())}], "shape": {"type": "full_image"}}],
        )
        db_session.add(db_dataset_item)
        db_session.flush()

        media_list = fxt_media_service.list_media(
            project_id=project.id,
            filters=MediaFilters(annotation_status=annotation_status),
        )

        videos = [m for m in media_list if isinstance(m, Video)]
        assert len(videos) == 1
        assert videos[0].annotated_frame_count == 1

    @pytest.mark.parametrize("subset", ["unassigned", "training"])
    def test_list_media_with_annotated_frame_subsets(
        self,
        subset: str,
        fxt_media_service: MediaService,
        fxt_project_with_media: tuple[Project, list[MediaDB]],
        db_session: Session,
    ) -> None:
        """Test listing media should include the video that has annotated frame with each filter option."""
        project, db_media_list = fxt_project_with_media

        db_video_frame = next(db_media for db_media in db_media_list if db_media.type == MediaType.VIDEO_FRAME)
        db_dataset_item = DatasetItemDB(
            id=db_video_frame.id,
            project_id=str(project.id),
            subset=subset,
            annotation_data=[{"labels": [{"id": str(uuid4())}], "shape": {"type": "full_image"}}],
        )
        db_session.add(db_dataset_item)
        db_session.flush()

        media_list = fxt_media_service.list_media(
            project_id=project.id,
            filters=MediaFilters(subsets=[subset]),
        )

        videos = [m for m in media_list if isinstance(m, Video)]
        assert len(videos) == 1
        assert videos[0].annotated_frame_count == 1

    @pytest.mark.parametrize(
        "annotation_status, expected_video_count",
        [
            (None, 1),
            (DatasetItemAnnotationStatus.WITH_ANNOTATIONS, 1),
            (DatasetItemAnnotationStatus.MISSING_ANNOTATIONS, 0),
        ],
    )
    def test_list_media_with_annotated_video(
        self,
        annotation_status: DatasetItemAnnotationStatus | None,
        expected_video_count: int,
        fxt_media_service: MediaService,
        fxt_project_with_pipeline: tuple[Project, Pipeline],
        fxt_media_factory: Callable[[str, list[dict]], list[MediaDB]],
        db_session: Session,
    ) -> None:
        """Test listing media with fully annotated video."""
        project, _ = fxt_project_with_pipeline

        video_config = {
            "id": str(uuid4()),
            "type": "video",
            "name": "test4",
            "format": "avi",
            "size": 1024,
            "width": 1024,
            "height": 768,
            "fps": 25.0,
            "frame_count": 1,
        }
        video_frame_config = {
            "video_id": video_config["id"],
            "type": "video_frame",
            "name": "test",
            "format": "jpg",
            "size": 1024,
            "width": 1024,
            "height": 768,
            "frame_index": 0,
        }

        db_media_list = fxt_media_factory(
            str(project.id),
            [
                video_config,
                video_frame_config,
            ],
        )

        db_video_frame = db_media_list[1]
        db_dataset_item = DatasetItemDB(
            id=db_video_frame.id,
            project_id=str(project.id),
            subset="unassigned",
            annotation_data=[{"labels": [{"id": str(uuid4())}], "shape": {"type": "full_image"}}],
        )
        db_session.add(db_dataset_item)
        db_session.flush()

        media_list = fxt_media_service.list_media(
            project_id=project.id,
            filters=MediaFilters(annotation_status=annotation_status),
            exclude_types=[MediaType.VIDEO_FRAME],
        )

        assert len(media_list) == expected_video_count

    @pytest.mark.parametrize(
        "exclude_types, count", [([MediaType.IMAGE], 2), ([MediaType.VIDEO], 4), ([MediaType.VIDEO_FRAME], 4)]
    )
    def test_list_media_excluding(
        self,
        fxt_media_service: MediaService,
        fxt_project_with_media: tuple[Project, list[MediaDB]],
        exclude_types: list[MediaType],
        count: int,
    ) -> None:
        """Test listing media with media types exclusion."""
        project, db_media_list = fxt_project_with_media

        media_list = fxt_media_service.list_media(
            project_id=project.id,
            exclude_types=exclude_types,
        )

        assert len(media_list) == count

    @pytest.mark.parametrize(
        "sort_direction, expected_order",
        [
            (None, ["test3", "test2", "test1"]),
            (SortDirection.DESC, ["test3", "test2", "test1"]),
            (SortDirection.ASC, ["test1", "test2", "test3"]),
        ],
    )
    def test_list_media_sort_direction(
        self,
        fxt_media_service: MediaService,
        fxt_project_with_media_at_distinct_dates: tuple[Project, list[MediaDB]],
        sort_direction: SortDirection | None,
        expected_order: list[str],
    ) -> None:
        """Test listing media ordered by upload date, ascending and descending."""
        project, _ = fxt_project_with_media_at_distinct_dates

        filters = MediaFilters() if sort_direction is None else MediaFilters(sort_direction=sort_direction)
        media_list = fxt_media_service.list_media(project_id=project.id, filters=filters)

        assert [media.name for media in media_list] == expected_order

    def test_list_media_sort_by_and_sort_direction_combined(
        self,
        fxt_media_service: MediaService,
        fxt_project_with_media_at_distinct_dates: tuple[Project, list[MediaDB]],
    ) -> None:
        """Test sorting by specified sort_by and sort_direction."""
        project, _ = fxt_project_with_media_at_distinct_dates

        media_list = fxt_media_service.list_media(
            project_id=project.id,
            filters=MediaFilters(sort_by=MediaSortBy.UPLOAD_DATE, sort_direction=SortDirection.ASC),
        )

        assert [media.name for media in media_list] == ["test1", "test2", "test3"]

    def test_get_media_by_id(
        self,
        fxt_media_service: MediaService,
        fxt_project_with_media: tuple[Project, list[MediaDB]],
    ):
        """Test retrieving a media by ID."""
        project, db_media_list = fxt_project_with_media

        fetched_media = fxt_media_service.get_media_by_id(project_id=project.id, media_id=UUID(db_media_list[0].id))

        assert str(fetched_media.id) == db_media_list[0].id and fetched_media.name == db_media_list[0].name

    def test_get_media_by_id_with_annotated_frame(
        self,
        fxt_media_service: MediaService,
        fxt_project_with_media: tuple[Project, list[MediaDB]],
        db_session: Session,
    ) -> None:
        """Test get video with annotated frame."""
        project, db_media_list = fxt_project_with_media

        db_video_frame = next(db_media for db_media in db_media_list if db_media.type == MediaType.VIDEO_FRAME)
        db_dataset_item = DatasetItemDB(
            id=db_video_frame.id,
            project_id=str(project.id),
            subset="unassigned",
            annotation_data=[{"labels": [{"id": str(uuid4())}], "shape": {"type": "full_image"}}],
        )
        db_session.add(db_dataset_item)
        db_session.flush()

        db_video = next(db_media for db_media in db_media_list if db_media.type == MediaType.VIDEO)

        media = fxt_media_service.get_media_by_id(project_id=project.id, media_id=UUID(db_video.id))

        assert isinstance(media, Video)
        assert media.annotated_frame_count == 1

    def test_get_media_by_id_not_found(
        self,
        fxt_media_service: MediaService,
        fxt_project_with_media: tuple[Project, list[MediaDB]],
    ):
        """Test retrieving a non-existent media raises error."""
        project, db_media_list = fxt_project_with_media
        non_existent_id = uuid4()

        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_media_service.get_media_by_id(project_id=project.id, media_id=non_existent_id)

        assert excinfo.value.resource_type == ResourceType.MEDIA
        assert excinfo.value.resource_id == str(non_existent_id)

    def test_get_media_by_ids(
        self,
        fxt_media_service: MediaService,
        fxt_project_with_media: tuple[Project, list[MediaDB]],
    ):
        """Test retrieving multiple media items by their IDs."""
        project, db_media_list = fxt_project_with_media
        media_ids = [UUID(media.id) for media in db_media_list[:2]]

        fetched_media = fxt_media_service.get_media_by_ids(project_id=project.id, media_ids=media_ids)
        assert len(fetched_media) == 2
        assert sorted([media.id for media in fetched_media]) == sorted(media_ids)

    def test_get_media_by_ids_not_found(
        self,
        fxt_media_service: MediaService,
        fxt_project_with_media: tuple[Project, list[MediaDB]],
    ):
        """Test retrieving multiple media items by non-existent IDs."""
        project, db_media_list = fxt_project_with_media
        non_existent_id = uuid4()

        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_media_service.get_media_by_ids(project_id=project.id, media_ids=[non_existent_id])
        assert excinfo.value.resource_type == ResourceType.MEDIA
        assert excinfo.value.resource_id == str(non_existent_id)

    def test_get_media_binary_path_by_id(
        self,
        tmp_path: Path,
        fxt_media_service: MediaService,
        fxt_project_with_media: tuple[Project, list[MediaDB]],
    ):
        """Test retrieving a media binary path by ID."""
        project, db_media_list = fxt_project_with_media

        media_binary_path = fxt_media_service.get_media_binary_path_by_id(
            project_id=project.id, media_id=UUID(db_media_list[0].id)
        )

        assert (
            media_binary_path
            == tmp_path / f"projects/{str(project.id)}/dataset/{db_media_list[0].id}.{db_media_list[0].format}"
        )

    def test_get_media_binary_path_by_id_not_found(
        self,
        fxt_media_service: MediaService,
        fxt_project_with_media: tuple[Project, list[MediaDB]],
    ):
        """Test retrieving a non-existent media binary path raises error."""
        project, db_media_list = fxt_project_with_media
        non_existent_id = uuid4()

        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_media_service.get_media_binary_path_by_id(project_id=project.id, media_id=non_existent_id)

        assert excinfo.value.resource_type == ResourceType.MEDIA
        assert excinfo.value.resource_id == str(non_existent_id)

    def test_get_media_thumbnail_path(
        self,
        tmp_path: Path,
        fxt_media_service: MediaService,
        fxt_project_with_media: tuple[Project, list[MediaDB]],
    ):
        """Test retrieving a media thumbnail path by ID."""
        project, db_media_list = fxt_project_with_media

        media_binary_path = fxt_media_service.get_media_thumbnail_path(project=project, media=db_media_list[0])

        assert media_binary_path == tmp_path / f"projects/{str(project.id)}/dataset/{db_media_list[0].id}-thumb.jpg"

    def test_delete_media(
        self,
        fxt_media_service: MediaService,
        fxt_project_with_media: tuple[Project, list[MediaDB]],
        fxt_projects_dir: Path,
        db_session: Session,
    ):
        """Test deleting a media."""
        project, db_media_list = fxt_project_with_media

        dataset_dir = fxt_projects_dir / str(project.id) / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        binary_path = dataset_dir / f"{db_media_list[0].id}.{db_media_list[0].format}"
        binary_path.touch()
        assert os.path.exists(binary_path)

        thumbnail_path = dataset_dir / f"{db_media_list[0].id}-thumb.jpg"
        thumbnail_path.touch()
        assert os.path.exists(thumbnail_path)

        """Test deleting a media."""
        fxt_media_service.delete_media(project=project, media_id=UUID(db_media_list[0].id))

        assert db_session.get(MediaDB, db_media_list[0].id) is None
        assert not os.path.exists(binary_path)
        assert not os.path.exists(thumbnail_path)

    def test_delete_media_not_found(
        self,
        fxt_media_service: MediaService,
        fxt_project_with_media: tuple[Project, list[MediaDB]],
    ):
        """Test deleting a non-existent media raises error."""
        project, db_media_list = fxt_project_with_media

        non_existent_id = uuid4()
        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_media_service.delete_media(project=project, media_id=non_existent_id)

        assert excinfo.value.resource_type == ResourceType.MEDIA
        assert excinfo.value.resource_id == str(non_existent_id)

    @pytest.mark.parametrize(
        "annotation_status, expected_count",
        [
            (None, 5),  # All items
            (DatasetItemAnnotationStatus.MISSING_ANNOTATIONS, 2),  # 2 items without annotations
            (DatasetItemAnnotationStatus.WITH_ANNOTATIONS, 3),  # 3 items with annotations
        ],
    )
    def test_count_media_with_annotation_status(
        self,
        fxt_media_service: MediaService,
        fxt_project_with_annotation_status_items: tuple[Project, list[DatasetItemDB]],
        annotation_status: DatasetItemAnnotationStatus | None,
        expected_count: int,
    ) -> None:
        """Test counting media with annotation_status filter."""
        project, db_dataset_items = fxt_project_with_annotation_status_items

        count = fxt_media_service.count_media(project=project, annotation_status=annotation_status)

        assert count == expected_count

    @pytest.mark.parametrize(
        "annotation_status, expected_names",
        [
            (None, ["unannotated1", "unannotated2", "reviewed1", "reviewed2", "reviewed3"]),
            (DatasetItemAnnotationStatus.MISSING_ANNOTATIONS, ["unannotated1", "unannotated2"]),
            (DatasetItemAnnotationStatus.WITH_ANNOTATIONS, ["reviewed1", "reviewed2", "reviewed3"]),
        ],
    )
    def test_list_media_with_annotation_status(
        self,
        fxt_media_service: MediaService,
        fxt_project_with_annotation_status_items: tuple[Project, list[DatasetItemDB]],
        annotation_status: DatasetItemAnnotationStatus | None,
        expected_names: list[str],
    ) -> None:
        """Test listing media with annotation_status filter."""
        project, db_dataset_items = fxt_project_with_annotation_status_items

        media_list = fxt_media_service.list_media(
            project_id=project.id,
            filters=MediaFilters(
                annotation_status=annotation_status,
            ),
        )

        assert len(media_list) == len(expected_names)
        actual_names = sorted([media.name for media in media_list])
        assert actual_names == sorted(expected_names)

    @pytest.mark.parametrize(
        "annotation_status, limit, offset, expected_count",
        [
            (DatasetItemAnnotationStatus.MISSING_ANNOTATIONS, 1, 0, 1),  # First page of unannotated
            (DatasetItemAnnotationStatus.MISSING_ANNOTATIONS, 1, 1, 1),  # Second page of unannotated
            (DatasetItemAnnotationStatus.MISSING_ANNOTATIONS, 1, 2, 0),  # Beyond available unannotated items
            (DatasetItemAnnotationStatus.WITH_ANNOTATIONS, 2, 0, 2),  # First page of reviewed
            (DatasetItemAnnotationStatus.WITH_ANNOTATIONS, 2, 2, 1),  # Second page of reviewed (only 1 left)
            (None, 10, 0, 5),  # All items
        ],
    )
    def test_list_media_with_annotation_status_pagination(
        self,
        fxt_media_service: MediaService,
        fxt_project_with_annotation_status_items: tuple[Project, list[DatasetItemDB]],
        annotation_status: DatasetItemAnnotationStatus | None,
        limit: int,
        offset: int,
        expected_count: int,
    ) -> None:
        """Test listing media with annotation_status filter and pagination."""
        project, db_dataset_items = fxt_project_with_annotation_status_items

        media_list = fxt_media_service.list_media(
            project_id=project.id,
            filters=MediaFilters(
                limit=limit,
                offset=offset,
                annotation_status=annotation_status,
            ),
        )

        assert len(media_list) == expected_count

    def test_list_media_annotation_status_combined_with_dates(
        self,
        fxt_media_service: MediaService,
        fxt_project_with_annotation_status_items: tuple[Project, list[DatasetItemDB]],
    ) -> None:
        """Test annotation_status filter combined with date filters."""
        project, db_dataset_items = fxt_project_with_annotation_status_items

        # All reviewed items within date range
        media_list = fxt_media_service.list_media(
            project_id=project.id,
            filters=MediaFilters(
                start_date=datetime.fromisoformat("2025-01-01T00:00:00Z"),
                end_date=datetime.fromisoformat("2025-02-02T00:00:00Z"),
                annotation_status=DatasetItemAnnotationStatus.WITH_ANNOTATIONS,
            ),
        )
        assert len(media_list) == 3

        # No items outside date range
        media_list = fxt_media_service.list_media(
            project_id=project.id,
            filters=MediaFilters(
                start_date=datetime.fromisoformat("2025-03-01T00:00:00Z"),
                end_date=datetime.fromisoformat("2025-03-31T00:00:00Z"),
                annotation_status=DatasetItemAnnotationStatus.MISSING_ANNOTATIONS,
            ),
        )
        assert len(media_list) == 0

    def test_annotation_status_filter_verifies_data_correctness(
        self,
        fxt_media_service: MediaService,
        fxt_project_with_annotation_status_items: tuple[Project, list[DatasetItemDB]],
    ) -> None:
        """Test that annotation_status filter returns items with correct properties."""
        project, db_dataset_items = fxt_project_with_annotation_status_items

        unannotated_items = fxt_media_service.list_media(
            project_id=project.id,
            filters=MediaFilters(
                annotation_status=DatasetItemAnnotationStatus.MISSING_ANNOTATIONS,
            ),
        )
        assert len(unannotated_items) == 2

        reviewed_media = fxt_media_service.list_media(
            project_id=project.id,
            filters=MediaFilters(
                annotation_status=DatasetItemAnnotationStatus.WITH_ANNOTATIONS,
            ),
        )
        assert len(reviewed_media) == 3

    def test_list_media_filter_by_single_label(
        self,
        fxt_media_service: MediaService,
        fxt_project_with_labeled_dataset_items: tuple[Project, list[DatasetItemDB]],
    ):
        """Test listing media filtered by a single label."""
        project, db_dataset_items = fxt_project_with_labeled_dataset_items
        label_0_id = project.task.labels[0].id

        # Filter by label_0 - should return media 1 and 3 (item_label_0 and item_both_labels)
        media_list = fxt_media_service.list_media(
            project_id=project.id,
            filters=MediaFilters(
                label_ids=[label_0_id],
            ),
        )

        assert len(media_list) == 2
        item_names = {item.name for item in media_list}
        assert item_names == {"item_label_0", "item_both_labels"}

    def test_list_media_filter_by_multiple_labels(
        self,
        fxt_media_service: MediaService,
        fxt_project_with_labeled_dataset_items: tuple[Project, list[DatasetItemDB]],
    ):
        """Test listing media filtered by multiple labels (OR logic)."""
        project, db_dataset_items = fxt_project_with_labeled_dataset_items
        label_0_id = project.task.labels[0].id
        label_1_id = project.task.labels[1].id

        # Filter by label_0 OR label_1 - should return items 1, 2, and 3
        media_list = fxt_media_service.list_media(
            project_id=project.id,
            filters=MediaFilters(
                label_ids=[label_0_id, label_1_id],
            ),
        )

        assert len(media_list) == 3
        item_names = {item.name for item in media_list}
        assert item_names == {"item_label_0", "item_label_1", "item_both_labels"}

    def test_list_media_filter_by_nonexistent_label(
        self,
        fxt_media_service: MediaService,
        fxt_project_with_labeled_dataset_items: tuple[Project, list[DatasetItemDB]],
    ):
        """Test listing media filtered by a nonexistent label."""
        project, db_dataset_items = fxt_project_with_labeled_dataset_items
        nonexistent_label_id = uuid4()

        # Filter by nonexistent label - should return empty list
        media_list = fxt_media_service.list_media(
            project_id=project.id,
            filters=MediaFilters(
                label_ids=[nonexistent_label_id],
            ),
        )

        assert len(media_list) == 0

    def test_count_media_filter_by_single_label(
        self,
        fxt_media_service: MediaService,
        fxt_project_with_labeled_dataset_items: tuple[Project, list[DatasetItemDB]],
    ):
        """Test counting media filtered by a single label."""
        project, db_dataset_items = fxt_project_with_labeled_dataset_items
        label_0_id = project.task.labels[0].id

        # Count media with label_0 - should return 2
        count = fxt_media_service.count_media(
            project=project,
            label_ids=[label_0_id],
        )

        assert count == 2

    def test_count_media_filter_by_multiple_labels(
        self,
        fxt_media_service: MediaService,
        fxt_project_with_labeled_dataset_items: tuple[Project, list[DatasetItemDB]],
    ):
        """Test counting media filtered by multiple labels (OR logic)."""
        project, db_dataset_items = fxt_project_with_labeled_dataset_items
        label_0_id = project.task.labels[0].id
        label_1_id = project.task.labels[1].id

        # Count media with label_0 OR label_1 - should return 3
        count = fxt_media_service.count_media(
            project=project,
            label_ids=[label_0_id, label_1_id],
        )

        assert count == 3

    def test_list_media_no_label_filter(
        self,
        fxt_media_service: MediaService,
        fxt_project_with_labeled_dataset_items: tuple[Project, list[DatasetItemDB]],
    ):
        """Test listing media without label filter returns all items."""
        project, db_dataset_items = fxt_project_with_labeled_dataset_items

        # No filter - should return all 4 items
        media_list = fxt_media_service.list_media(project_id=project.id)

        assert len(media_list) == 4
        item_names = {item.name for item in media_list}
        assert item_names == {"item_no_labels", "item_label_0", "item_label_1", "item_both_labels"}

    def test_list_media_filter_by_label_returns_video_not_frames(
        self,
        fxt_media_service: MediaService,
        fxt_project_with_labeled_video: tuple[Project, MediaDB, list[MediaDB]],
    ):
        """Filtering by a label annotated only on video frames returns the parent video, not the frames."""
        project, video, frames = fxt_project_with_labeled_video
        label_0_id = project.task.labels[0].id

        # Mirror the router behaviour of excluding raw video frames from the listing
        media_list = fxt_media_service.list_media(
            project_id=project.id,
            filters=MediaFilters(label_ids=[label_0_id]),
            exclude_types=[MediaType.VIDEO_FRAME],
        )

        assert len(media_list) == 1
        returned = media_list[0]
        assert str(returned.id) == video.id
        assert returned.type == MediaType.VIDEO
        # Frames must not be returned individually
        returned_ids = {str(item.id) for item in media_list}
        assert all(frame.id not in returned_ids for frame in frames)

    def test_count_media_filter_by_label_counts_video_once(
        self,
        fxt_media_service: MediaService,
        fxt_project_with_labeled_video: tuple[Project, MediaDB, list[MediaDB]],
    ):
        """A video with multiple frames carrying the label is counted once, not per frame."""
        project, video, frames = fxt_project_with_labeled_video
        label_0_id = project.task.labels[0].id

        count = fxt_media_service.count_media(
            project=project,
            label_ids=[label_0_id],
            exclude_types=[MediaType.VIDEO_FRAME],
        )

        assert count == 1

    def test_list_media_filter_by_label_returns_image_only(
        self,
        fxt_media_service: MediaService,
        fxt_project_with_labeled_video: tuple[Project, MediaDB, list[MediaDB]],
    ):
        """Filtering by a label present only on a standalone image returns that image, not the video."""
        project, video, frames = fxt_project_with_labeled_video
        label_1_id = project.task.labels[1].id

        media_list = fxt_media_service.list_media(
            project_id=project.id,
            filters=MediaFilters(label_ids=[label_1_id]),
            exclude_types=[MediaType.VIDEO_FRAME],
        )

        assert len(media_list) == 1
        assert media_list[0].name == "labeled_image"
        assert media_list[0].type == MediaType.IMAGE

    @pytest.mark.parametrize(
        "subset, expected_count",
        [
            (None, 8),  # All items
            (["unassigned"], 2),  # 2 unassigned items
            (["training"], 3),  # 3 training items
            (["validation"], 2),  # 2 validation items
            (["testing"], 1),  # 1 testing item
            (["training", "validation"], 5),  # 3 training + 2 validation items
        ],
    )
    def test_count_media_with_subset(
        self,
        fxt_media_service: MediaService,
        fxt_project_with_subset_items: tuple[Project, list[DatasetItemDB]],
        subset: list[str] | None,
        expected_count: int,
    ) -> None:
        """Test counting media with subset filter."""
        project, db_dataset_items = fxt_project_with_subset_items

        count = fxt_media_service.count_media(project=project, subsets=subset)

        assert count == expected_count

    @pytest.mark.parametrize(
        "subset, expected_names",
        [
            (
                None,
                [
                    "unassigned1",
                    "unassigned2",
                    "training1",
                    "training2",
                    "training3",
                    "validation1",
                    "validation2",
                    "testing1",
                ],
            ),
            (["unassigned"], ["unassigned1", "unassigned2"]),
            (["training"], ["training1", "training2", "training3"]),
            (["validation"], ["validation1", "validation2"]),
            (["testing"], ["testing1"]),
            (["validation", "testing"], ["validation1", "validation2", "testing1"]),
        ],
    )
    def test_list_media_with_subset(
        self,
        fxt_media_service: MediaService,
        fxt_project_with_subset_items: tuple[Project, list[DatasetItemDB]],
        subset: list[str] | None,
        expected_names: list[str],
    ) -> None:
        """Test listing media with subset filter."""
        project, db_dataset_items = fxt_project_with_subset_items

        media_list = fxt_media_service.list_media(
            project_id=project.id,
            filters=MediaFilters(
                limit=20,
                offset=0,
                subsets=subset,
            ),
        )

        assert len(media_list) == len(expected_names)
        actual_names = sorted([item.name for item in media_list])
        assert actual_names == sorted(expected_names)

    @pytest.mark.parametrize(
        "subset, limit, offset, expected_count",
        [
            (["unassigned"], 1, 0, 1),  # First page of unassigned
            (["unassigned"], 1, 1, 1),  # Second page of unassigned
            (["unassigned"], 1, 2, 0),  # Beyond available unassigned items
            (["training"], 2, 0, 2),  # First page of training
            (["training"], 2, 2, 1),  # Second page of training (only 1 left)
            (["validation"], 10, 0, 2),  # All validation items
            (["testing"], 10, 0, 1),  # All testing items
        ],
    )
    def test_list_media_with_subset_pagination(
        self,
        fxt_media_service: MediaService,
        fxt_project_with_subset_items: tuple[Project, list[DatasetItemDB]],
        subset: list[str] | None,
        limit: int,
        offset: int,
        expected_count: int,
    ) -> None:
        """Test listing media with subset filter and pagination."""
        project, db_dataset_items = fxt_project_with_subset_items

        media_list = fxt_media_service.list_media(
            project_id=project.id,
            filters=MediaFilters(
                limit=limit,
                offset=offset,
                subsets=subset,
            ),
        )

        assert len(media_list) == expected_count

    def test_subset_filter_verifies_data_correctness(
        self,
        fxt_media_service: MediaService,
        fxt_project_with_subset_items: tuple[Project, list[DatasetItemDB]],
    ) -> None:
        """Test that subset filter returns media with correct subset values."""
        project, db_dataset_items = fxt_project_with_subset_items

        # Unassigned items should have subset=unassigned
        unassigned_items = fxt_media_service.list_media(
            project_id=project.id,
            filters=MediaFilters(
                limit=20,
                offset=0,
                subsets=["unassigned"],
            ),
        )
        assert len(unassigned_items) == 2

        # Training items should have subset=training
        training_items = fxt_media_service.list_media(
            project_id=project.id,
            filters=MediaFilters(
                limit=20,
                offset=0,
                subsets=["training"],
            ),
        )
        assert len(training_items) == 3

        # Validation items should have subset=validation
        validation_items = fxt_media_service.list_media(
            project_id=project.id,
            filters=MediaFilters(
                limit=20,
                offset=0,
                subsets=["validation"],
            ),
        )
        assert len(validation_items) == 2

        # Testing items should have subset=testing
        testing_items = fxt_media_service.list_media(
            project_id=project.id,
            filters=MediaFilters(
                limit=20,
                offset=0,
                subsets=["testing"],
            ),
        )
        assert len(testing_items) == 1

        # Multiple subsets should return the union of matching items
        training_and_testing_items = fxt_media_service.list_media(
            project_id=project.id,
            filters=MediaFilters(
                limit=20,
                offset=0,
                subsets=["training", "testing"],
            ),
        )
        assert len(training_and_testing_items) == 4

    def test_save_video_frame(
        self,
        tmp_path: Path,
        fxt_video_data: Callable[[Path], None],
        fxt_projects_dir: Path,
        fxt_media_service: MediaService,
        fxt_project_with_media: tuple[Project, list[MediaDB]],
        db_session: Session,
    ):
        """Test saving a videoframe."""
        project, db_media_list = fxt_project_with_media
        media = Video.model_validate(db_media_list[3], from_attributes=True)

        # Create the dataset directory
        dataset_dir = tmp_path / fxt_projects_dir / str(project.id) / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        frame_image = PILImage.new("RGB", (640, 480))

        video_frame = fxt_media_service.save_video_frame(
            project=project, video=media, frame_index=50, frame_image=frame_image
        )

        video_frame_binary_path = dataset_dir / f"{video_frame.id}.jpg"
        assert os.path.exists(video_frame_binary_path)

        video_frame_thumbnail_path = dataset_dir / f"{video_frame.id}-thumb.jpg"
        assert os.path.exists(video_frame_thumbnail_path)

        db_video_frame = db_session.get(MediaDB, str(video_frame.id))
        assert db_video_frame is not None
        assert (
            db_video_frame.id == str(video_frame.id)
            and db_video_frame.project_id == str(project.id)
            and db_video_frame.type == "video_frame"
            and db_video_frame.name == "test4_frame_50"
            and db_video_frame.format == "jpg"
            and db_video_frame.width == 640
            and db_video_frame.height == 480
            and db_video_frame.video_id == str(media.id)
            and db_video_frame.frame_index == 50
        )

    def test_get_video_frame_by_video_id_and_index(
        self,
        fxt_media_service: MediaService,
        fxt_video_frame: Callable[[float], tuple[DatasetItemDB, MediaDB]],
        fxt_project_with_media: tuple[Project, list[MediaDB]],
    ) -> None:
        """Test getting a video frame by video ID and index."""
        project, db_media_list = fxt_project_with_media
        media = db_media_list[3]
        fxt_video_frame(250)

        video_frame = fxt_media_service.get_video_frame_by_video_id_and_index(
            project=project, video_id=UUID(media.id), frame_index=250
        )
        assert video_frame is not None

    def test_get_non_existing_video_frame_by_video_id_and_index(
        self,
        fxt_media_service: MediaService,
        fxt_video_frame: Callable[[float], tuple[DatasetItemDB, MediaDB]],
        fxt_project_with_media: tuple[Project, list[MediaDB]],
    ) -> None:
        """Test getting a non extracted video frame by video ID and index."""
        project, db_media_list = fxt_project_with_media
        media = db_media_list[3]
        fxt_video_frame(250)

        video_frame = fxt_media_service.get_video_frame_by_video_id_and_index(
            project=project, video_id=UUID(media.id), frame_index=500
        )
        assert video_frame is None

    def test_search_video_frames_by_video_id_and_indexes(
        self,
        fxt_media_service: MediaService,
        fxt_video_frame: Callable[[float], tuple[DatasetItemDB, MediaDB]],
        fxt_project_with_media: tuple[Project, list[MediaDB]],
    ) -> None:
        """Test searching video frames by video ID and indexes."""
        project, db_media_list = fxt_project_with_media
        media = db_media_list[3]
        fxt_video_frame(250)
        fxt_video_frame(260)
        fxt_video_frame(270)

        video_frames = fxt_media_service.search_video_frames_by_video_id_and_indexes(
            project=project, video_id=UUID(media.id), frame_indexes=[250, 260, 280]
        )
        assert video_frames is not None
        assert len(video_frames) == 2
        assert sorted([video_frame.frame_index for video_frame in video_frames]) == [250, 260]

    def test_get_video_frames_by_video_id(
        self,
        fxt_media_service: MediaService,
        fxt_video_frame: Callable[[float], tuple[DatasetItemDB, MediaDB]],
        fxt_project_with_media: tuple[Project, list[MediaDB]],
    ) -> None:
        """Test getting a list of video frames by video ID."""
        project, db_media_list = fxt_project_with_media
        media = db_media_list[3]
        fxt_video_frame(1.0)
        fxt_video_frame(2.0)

        video_frames = fxt_media_service.list_annotated_video_frames_by_video_id(
            project=project, video_id=UUID(media.id)
        )
        assert video_frames is not None
        assert len(video_frames) == 2

    @pytest.mark.parametrize("index_from, index_to", [(0, 35), (35, 55)])
    def test_get_video_frames_by_video_id_range(
        self,
        index_from: int,
        index_to: int,
        fxt_media_service: MediaService,
        fxt_video_frame: Callable[[float], tuple[DatasetItemDB, MediaDB]],
        fxt_project_with_media: tuple[Project, list[MediaDB]],
    ) -> None:
        """Test getting a list of video frames by video ID."""
        project, db_media_list = fxt_project_with_media
        media = db_media_list[3]
        fxt_video_frame(30.0)
        fxt_video_frame(40.0)

        video_frames = fxt_media_service.list_annotated_video_frames_by_video_id(
            project=project, video_id=UUID(media.id), frame_index_from=index_from, frame_index_to=index_to
        )
        assert video_frames is not None
        assert len(video_frames) == 1
