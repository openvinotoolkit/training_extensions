# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from datetime import datetime
from pathlib import Path
from uuid import UUID, uuid4

import cv2
import datumaro.experimental as dm
import numpy as np
import polars as pl
import pytest
from PIL import Image
from sqlalchemy.orm import Session

from app.datumaro_converter import SampleMode
from app.db.schema import DatasetItemDB, DatasetRevisionDB, MediaDB, PipelineDB
from app.models import DatasetItemAnnotationStatus, DatasetItemSubset, Pipeline, Project
from app.services import DatasetRevisionService, DatasetService, DatasetViewService, MediaService
from app.services.base import ResourceNotFoundError, ResourceType
from app.services.dataset_revision_service import DATASET_REVISION_ITEM_THUMBNAIL_SIZE


@pytest.fixture
def fxt_dataset_revision_service(
    fxt_projects_dir: Path,
    db_session: Session,
) -> DatasetRevisionService:
    """Fixture to create a DatasetRevisionService instance."""
    return DatasetRevisionService(data_dir=fxt_projects_dir.parent, db_session=db_session)


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

    return fxt_project_service.get_project_by_id(UUID(db_project.id)), fxt_pipeline_service.get_pipeline_by_id(
        UUID(db_project.id)
    )


@pytest.fixture
def fxt_project_with_subset_items(
    fxt_project_with_pipeline, db_session
) -> tuple[Project, list[tuple[MediaDB, DatasetItemDB]]]:
    """Fixture with dataset items covering all subset types."""
    project, _ = fxt_project_with_pipeline

    distribution = [
        (DatasetItemSubset.UNASSIGNED, 2),
        (DatasetItemSubset.TRAINING, 3),
        (DatasetItemSubset.VALIDATION, 2),
        (DatasetItemSubset.TESTING, 1),
    ]

    db_media_and_dataset_items = []
    for subset, item_count in distribution:
        for idx in range(item_count):
            db_media = MediaDB(
                type="image",
                name=f"{subset.value}{idx + 1}",
                format="jpg",
                size=1024,
                width=1024,
                height=768,
                project_id=str(project.id),
                created_at=datetime.fromisoformat("2025-02-01T00:00:00Z"),
            )
            db_session.add(db_media)
            db_session.flush()

            dataset_item = DatasetItemDB(
                subset=subset,
                user_reviewed=False,
                project_id=str(project.id),
                created_at=datetime.fromisoformat("2025-02-01T00:00:00Z"),
                updated_at=datetime.fromisoformat("2025-02-01T00:00:00Z"),
            )
            dataset_item.id = db_media.id

            db_session.add(dataset_item)
            db_session.flush()

            db_media_and_dataset_items.append((db_media, dataset_item))

    return project, db_media_and_dataset_items


@pytest.fixture
def fxt_project_with_subset_items_on_disk(
    fxt_projects_dir, fxt_db_labels, fxt_project_with_pipeline, db_session, fxt_label_service
) -> tuple[Project, list[tuple[MediaDB, DatasetItemDB]]]:
    """Fixture with dataset items covering all subset types and annotation data."""
    project, _ = fxt_project_with_pipeline

    # Get the first label for annotation
    label = fxt_db_labels[0]
    label_id = str(label.id)

    def annotation():
        # Rectangle annotation with one label
        return [
            {
                "shape": {
                    "type": "rectangle",
                    "x": 10,
                    "y": 10,
                    "width": 100,
                    "height": 100,
                },
                "labels": [{"id": label_id, "name": label.name, "color": label.color}],
            }
        ]

    def make_item(name: str, subset: DatasetItemSubset, created_at: datetime):
        db_media = MediaDB(
            type="image",
            name=name,
            format="jpg",
            size=1024,
            width=1024,
            height=768,
            project_id=str(project.id),
            created_at=created_at,
        )
        db_session.add(db_media)
        db_session.flush()

        dataset_item = DatasetItemDB(
            subset=subset.name.lower(),
            user_reviewed=True,
            project_id=str(project.id),
            created_at=created_at,
            updated_at=created_at,
            annotation_data=annotation(),
        )
        dataset_item.id = db_media.id

        db_session.add(dataset_item)
        db_session.flush()

        return db_media, dataset_item

    media_and_dataset_items = [
        make_item("unassigned1", DatasetItemSubset.UNASSIGNED, datetime.fromisoformat("2025-02-01T00:00:00Z")),
        make_item("unassigned2", DatasetItemSubset.UNASSIGNED, datetime.fromisoformat("2025-02-02T00:00:00Z")),
        make_item("training1", DatasetItemSubset.TRAINING, datetime.fromisoformat("2025-02-03T00:00:00Z")),
        make_item("training2", DatasetItemSubset.TRAINING, datetime.fromisoformat("2025-02-04T00:00:00Z")),
        make_item("training3", DatasetItemSubset.TRAINING, datetime.fromisoformat("2025-02-05T00:00:00Z")),
        make_item("validation1", DatasetItemSubset.VALIDATION, datetime.fromisoformat("2025-02-06T00:00:00Z")),
        make_item("validation2", DatasetItemSubset.VALIDATION, datetime.fromisoformat("2025-02-07T00:00:00Z")),
        make_item("testing1", DatasetItemSubset.TESTING, datetime.fromisoformat("2025-02-08T00:00:00Z")),
    ]

    # Create images directory
    images_dir = fxt_projects_dir / str(project.id) / "dataset"
    images_dir.mkdir(parents=True, exist_ok=True)

    def create_item(item: MediaDB) -> None:
        # Create dummy image file
        image_path = images_dir / f"{item.id}.{item.format}"
        image_path.write_bytes(b"\x00")  # 1-byte dummy file

    for media_db, _ in media_and_dataset_items:
        create_item(media_db)

    return project, media_and_dataset_items


@pytest.fixture
def fxt_project_with_video_frame_items_on_disk(
    fxt_projects_dir, fxt_db_labels, fxt_project_with_pipeline, db_session, fxt_label_service
) -> tuple[Project, list[tuple[MediaDB, DatasetItemDB]], MediaDB]:
    """Fixture with video frame dataset items (video + frames with annotations) and media on disk."""
    project, _ = fxt_project_with_pipeline

    label = fxt_db_labels[0]
    label_id = str(label.id)

    def annotation():
        return [
            {
                "shape": {
                    "type": "rectangle",
                    "x": 10,
                    "y": 10,
                    "width": 100,
                    "height": 100,
                },
                "labels": [{"id": label_id, "name": label.name, "color": label.color}],
            }
        ]

    # Create the parent video media entry
    db_video = MediaDB(
        type="video",
        name="test_video",
        format="mp4",
        size=10240,
        width=1920,
        height=1080,
        fps=30.0,
        frame_count=100,
        project_id=str(project.id),
        created_at=datetime.fromisoformat("2025-02-01T00:00:00Z"),
    )
    db_session.add(db_video)
    db_session.flush()

    # Create video frame media entries referencing the parent video
    media_and_dataset_items: list[tuple[MediaDB, DatasetItemDB]] = []
    for idx, (subset, frame_index) in enumerate(
        [
            (DatasetItemSubset.TRAINING, 5),
            (DatasetItemSubset.TRAINING, 15),
            (DatasetItemSubset.VALIDATION, 30),
        ]
    ):
        db_frame = MediaDB(
            type="video_frame",
            name=f"frame_{frame_index}",
            format="jpg",
            size=1024,
            width=1920,
            height=1080,
            video_id=str(db_video.id),
            frame_index=frame_index,
            project_id=str(project.id),
            created_at=datetime.fromisoformat("2025-02-01T00:00:00Z"),
        )
        db_session.add(db_frame)
        db_session.flush()

        dataset_item = DatasetItemDB(
            subset=subset.name.lower(),
            user_reviewed=True,
            project_id=str(project.id),
            created_at=datetime.fromisoformat("2025-02-01T00:00:00Z"),
            updated_at=datetime.fromisoformat("2025-02-01T00:00:00Z"),
            annotation_data=annotation(),
        )
        dataset_item.id = db_frame.id
        db_session.add(dataset_item)
        db_session.flush()

        media_and_dataset_items.append((db_frame, dataset_item))

    # Create media files on disk
    images_dir = fxt_projects_dir / str(project.id) / "dataset"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Create the video file on disk (dummy)
    video_path = images_dir / f"{db_video.id}.{db_video.format}"
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (1920, 1080))
    # Write enough frames to cover max frame_index used (30)
    for _ in range(31):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()

    # Create the frame image files on disk (dummy)
    for db_frame, _ in media_and_dataset_items:
        frame_path = images_dir / f"{db_frame.id}.{db_frame.format}"
        frame_path.write_bytes(b"\x00")

    return project, media_and_dataset_items, db_video


@pytest.fixture
def fxt_dataset_revision_with_parquet(
    fxt_projects_dir: Path,
    fxt_project_with_pipeline: tuple[Project, Pipeline],
    db_session: Session,
) -> tuple[Project, UUID]:
    """Fixture that creates a dataset revision with a Parquet file and image."""
    project, _ = fxt_project_with_pipeline
    revision_id = uuid4()

    # Create revision in database
    db_revision = DatasetRevisionDB(
        id=str(revision_id),
        project_id=str(project.id),
        name=f"Dataset ({str(revision_id).split('-')[0]})",
        files_deleted=False,
    )
    db_session.add(db_revision)
    db_session.flush()

    # Create revision directory structure
    revision_path = fxt_projects_dir / str(project.id) / "dataset_revisions" / str(revision_id)
    images_path = revision_path / "images"
    images_path.mkdir(parents=True, exist_ok=True)

    # Create a real image file
    item_id = uuid4()
    image_filename = "img_000001.jpg"
    image_path = images_path / image_filename

    # Create a simple test image (64x64 red square)
    test_image = Image.new("RGB", (1024, 768), color="red")
    test_image.save(image_path, "JPEG")

    # Create Polars dataframe with expected schema
    df = pl.DataFrame(
        {
            "id": [str(item_id)],
            "image": [str(image_filename)],
            "image_info": [{"width": 1024, "height": 768}],
            "subset": ["TRAINING"],
        }
    )

    # Save as Parquet
    parquet_path = revision_path / "data.parquet"
    df.write_parquet(parquet_path)

    return project, revision_id


@pytest.fixture
def fxt_dataset_revision_with_16bit_parquet(
    fxt_projects_dir: Path,
    fxt_project_with_pipeline: tuple[Project, Pipeline],
    db_session: Session,
) -> tuple[Project, UUID]:
    """Fixture that creates a dataset revision whose image is a high-bit-depth (16-bit) TIFF.

    The pixel values are confined to a narrow band at the top of the 16-bit range
    (60000-65000). A naive ``.convert("RGB")`` keeps only the most-significant byte,
    which would map every pixel to a near-white value (washed-out thumbnail). Correct
    min-max normalization should instead spread the values across the full [0, 255] range.
    """
    project, _ = fxt_project_with_pipeline
    revision_id = uuid4()

    # Create revision in database
    db_revision = DatasetRevisionDB(
        id=str(revision_id),
        project_id=str(project.id),
        name=f"Dataset ({str(revision_id).split('-')[0]})",
        files_deleted=False,
    )
    db_session.add(db_revision)
    db_session.flush()

    # Create revision directory structure
    revision_path = fxt_projects_dir / str(project.id) / "dataset_revisions" / str(revision_id)
    images_path = revision_path / "images"
    images_path.mkdir(parents=True, exist_ok=True)

    # Create a real 16-bit image file with values confined to a narrow high band.
    item_id = uuid4()
    image_filename = "img_16bit.tiff"
    image_path = images_path / image_filename

    width, height = 1024, 768
    gradient = np.linspace(60000, 65000, num=width, dtype=np.uint16)
    arr = np.tile(gradient, (height, 1))
    high_bit_image = Image.fromarray(arr)  # mode I;16
    assert high_bit_image.mode in ("I;16", "I")  # sanity check the fixture is actually high bit depth
    high_bit_image.save(image_path, "TIFF")

    # Create Polars dataframe with expected schema
    df = pl.DataFrame(
        {
            "id": [str(item_id)],
            "image": [str(image_filename)],
            "image_info": [{"width": width, "height": height}],
            "subset": ["TRAINING"],
        }
    )

    # Save as Parquet
    parquet_path = revision_path / "data.parquet"
    df.write_parquet(parquet_path)

    return project, revision_id


class TestDatasetRevisionServiceIntegration:
    """Integration tests for DatasetRevisionService."""

    def test_get_dm_dataset_video_frames_training_mode(
        self,
        fxt_projects_dir: Path,
        fxt_dataset_service: DatasetService,
        fxt_project_with_video_frame_items_on_disk: tuple[Project, list[tuple[MediaDB, DatasetItemDB]], MediaDB],
    ) -> None:
        """Test that get_dm_dataset with video frames in TRAINING mode uses frame binary paths."""
        project, media_and_dataset_items, db_video = fxt_project_with_video_frame_items_on_disk

        dataset = fxt_dataset_service.get_dm_dataset(
            project_id=project.id,
            task=project.task,
            annotation_status=DatasetItemAnnotationStatus.WITH_ANNOTATIONS,
            sample_mode=SampleMode.TRAINING,
        )

        assert isinstance(dataset, dm.Dataset)
        # All 3 video frame items should be in the dataset
        assert len(dataset) == len(media_and_dataset_items)

        # In TRAINING mode, media paths should point to the individual frame files, not the video
        for idx, sample in enumerate(dataset):
            media_item = media_and_dataset_items[idx][0]
            media_path = str(fxt_projects_dir / str(project.id) / "dataset" / f"{media_item.id}.{media_item.format}")
            assert sample.image.path == media_path

    def test_get_dm_dataset_video_frames_import_export_mode(
        self,
        fxt_projects_dir: Path,
        fxt_dataset_service: DatasetService,
        fxt_project_with_video_frame_items_on_disk: tuple[Project, list[tuple[MediaDB, DatasetItemDB]], MediaDB],
    ) -> None:
        """Test that get_dm_dataset with video frames in IMPORT_EXPORT mode uses the video binary path."""
        project, media_and_dataset_items, db_video = fxt_project_with_video_frame_items_on_disk

        dataset = fxt_dataset_service.get_dm_dataset(
            project_id=project.id,
            task=project.task,
            annotation_status=DatasetItemAnnotationStatus.WITH_ANNOTATIONS,
            sample_mode=SampleMode.IMPORT_EXPORT,
        )

        assert isinstance(dataset, dm.Dataset)
        assert len(dataset) == len(media_and_dataset_items)

        # In IMPORT_EXPORT mode, media paths for video frames should point to the parent video file
        expected_video_path = str(fxt_projects_dir / str(project.id) / "dataset" / f"{db_video.id}.{db_video.format}")
        for sample in dataset:
            assert sample.media.video_path == expected_video_path

    def test_save_revision(
        self,
        fxt_projects_dir: Path,
        fxt_dataset_service: DatasetService,
        fxt_dataset_revision_service: DatasetRevisionService,
        fxt_project_with_subset_items_on_disk: tuple[Project, list[tuple[MediaDB, DatasetItemDB]]],
        db_session: Session,
    ) -> None:
        """Test saving a dataset revision."""
        project, _ = fxt_project_with_subset_items_on_disk
        dataset = fxt_dataset_service.get_dm_dataset(
            project.id, project.task, DatasetItemAnnotationStatus.WITH_ANNOTATIONS, SampleMode.TRAINING
        )

        revision_id = fxt_dataset_revision_service.save_revision(
            project_id=project.id,
            dataset=dataset,
        )

        # Verify that a revision entry was created
        dataset_revision = db_session.get(DatasetRevisionDB, str(revision_id))
        assert dataset_revision is not None
        assert not dataset_revision.files_deleted
        assert dataset_revision.total_count == 8
        assert 5000 < dataset_revision.size < (dataset_revision.total_count * 1024)  # assuming each item overhead ~ 1KB
        assert dataset_revision.training_count == 3
        assert dataset_revision.validation_count == 2
        assert dataset_revision.testing_count == 1
        assert (fxt_projects_dir / str(project.id) / "dataset_revisions" / str(revision_id) / "data.parquet").exists()

    def test_save_revision_zero_count(
        self,
        fxt_dataset_service: DatasetService,
        fxt_dataset_revision_service: DatasetRevisionService,
        fxt_project_with_subset_items: tuple[Project, list[tuple[MediaDB, DatasetItemDB]]],
    ) -> None:
        """Test saving a dataset revision."""
        project, _ = fxt_project_with_subset_items
        dataset = fxt_dataset_service.get_dm_dataset(
            project.id, project.task, DatasetItemAnnotationStatus.WITH_ANNOTATIONS, SampleMode.TRAINING
        )

        with pytest.raises(ValueError):
            fxt_dataset_revision_service.save_revision(
                project_id=project.id,
                dataset=dataset,
            )

    def test_get_dataset_revision(
        self,
        fxt_dataset_service: DatasetService,
        fxt_dataset_revision_service: DatasetRevisionService,
        fxt_project_with_subset_items_on_disk: tuple[Project, list[tuple[MediaDB, DatasetItemDB]]],
    ) -> None:
        """Test getting a dataset revision."""
        project, _ = fxt_project_with_subset_items_on_disk
        dataset = fxt_dataset_service.get_dm_dataset(
            project.id, project.task, DatasetItemAnnotationStatus.WITH_ANNOTATIONS, SampleMode.TRAINING
        )

        # Save a revision
        revision_id = fxt_dataset_revision_service.save_revision(
            project_id=project.id,
            dataset=dataset,
        )

        # Now get the revision
        revision = fxt_dataset_revision_service.get_dataset_revision(project_id=project.id, revision_id=revision_id)

        assert revision is not None
        assert revision.id == revision_id
        assert revision.name == f"Dataset ({str(revision.id).split('-')[0]})"
        assert revision.files_deleted is False

    def test_get_latest_uptodate_dataset_revision_success(
        self,
        fxt_dataset_service: DatasetService,
        fxt_dataset_revision_service: DatasetRevisionService,
        fxt_project_with_subset_items_on_disk: tuple[Project, list[tuple[MediaDB, DatasetItemDB]]],
    ) -> None:
        """Test getting a dataset revision."""
        project, _ = fxt_project_with_subset_items_on_disk
        dataset = fxt_dataset_service.get_dm_dataset(
            project.id, project.task, DatasetItemAnnotationStatus.WITH_ANNOTATIONS, SampleMode.TRAINING
        )

        # Save a revision
        revision_id = fxt_dataset_revision_service.save_revision(
            project_id=project.id,
            dataset=dataset,
        )

        # Now get the revision
        latest_revision = fxt_dataset_revision_service.get_latest_uptodate_dataset_revision(project_id=project.id)

        assert latest_revision is not None
        assert latest_revision.id == revision_id

    def test_get_latest_uptodate_dataset_revision_not_exists(
        self,
        fxt_dataset_service: DatasetService,
        fxt_dataset_revision_service: DatasetRevisionService,
        fxt_project_with_subset_items_on_disk: tuple[Project, list[tuple[MediaDB, DatasetItemDB]]],
    ) -> None:
        """Test getting a dataset revision."""
        project, media_and_dataset_items = fxt_project_with_subset_items_on_disk
        dataset = fxt_dataset_service.get_dm_dataset(
            project.id, project.task, DatasetItemAnnotationStatus.WITH_ANNOTATIONS, SampleMode.TRAINING
        )

        # Save a revision
        fxt_dataset_revision_service.save_revision(
            project_id=project.id,
            dataset=dataset,
        )

        # Update dataset item
        dataset_item = next(
            ds_item
            for _, ds_item in media_and_dataset_items
            if ds_item.subset == DatasetItemSubset.UNASSIGNED.name.lower()
        )
        fxt_dataset_service.assign_dataset_item_subset(
            project_id=project.id, dataset_item_id=UUID(dataset_item.id), subset=DatasetItemSubset.TRAINING
        )

        # Now get the revision
        latest_revision = fxt_dataset_revision_service.get_latest_uptodate_dataset_revision(project_id=project.id)

        assert latest_revision is None

    def test_get_latest_uptodate_dataset_revision_after_subset_assignment_same_second(
        self,
        fxt_dataset_service: DatasetService,
        fxt_dataset_revision_service: DatasetRevisionService,
        fxt_project_with_subset_items_on_disk: tuple[Project, list[tuple[MediaDB, DatasetItemDB]]],
    ) -> None:
        """Regression test for dataset duplication after training.

        Reproduces the scenario where unassigned items get a subset assigned right before a dataset revision is
        saved (as happens during training when annotations are filled in after import). The item's `updated_at` is
        stored with microsecond precision, while the revision's `created_at` was previously stored with
        whole-second precision (SQLite CURRENT_TIMESTAMP), truncated down. When both happen within the same second,
        the freshly saved revision was wrongly considered stale, causing a duplicate dataset on the next training.
        """
        project, media_and_dataset_items = fxt_project_with_subset_items_on_disk

        # Assign a subset to a previously-unassigned item; this bumps its `updated_at` to "now".
        dataset_item = next(
            ds_item
            for _, ds_item in media_and_dataset_items
            if ds_item.subset == DatasetItemSubset.UNASSIGNED.name.lower()
        )
        fxt_dataset_service.assign_dataset_item_subset(
            project_id=project.id, dataset_item_id=UUID(dataset_item.id), subset=DatasetItemSubset.TRAINING
        )

        # Save a revision immediately afterwards (same wall-clock second as the subset assignment).
        dataset = fxt_dataset_service.get_dm_dataset(
            project.id, project.task, DatasetItemAnnotationStatus.WITH_ANNOTATIONS, SampleMode.TRAINING
        )
        revision_id = fxt_dataset_revision_service.save_revision(
            project_id=project.id,
            dataset=dataset,
        )

        # The just-created revision must be recognized as up to date (no duplicate would be created on next training).
        latest_revision = fxt_dataset_revision_service.get_latest_uptodate_dataset_revision(project_id=project.id)

        assert latest_revision is not None
        assert latest_revision.id == revision_id

    def test_get_dataset_revision_not_found(
        self,
        fxt_dataset_revision_service: DatasetRevisionService,
        fxt_project_with_subset_items: tuple[Project, list[DatasetItemDB]],
    ) -> None:
        """Test getting a non-existent dataset revision raises error."""
        project, _ = fxt_project_with_subset_items
        non_existent_id = uuid4()

        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_dataset_revision_service.get_dataset_revision(project_id=project.id, revision_id=non_existent_id)

        assert excinfo.value.resource_type == ResourceType.DATASET_REVISION
        assert excinfo.value.resource_id == str(non_existent_id)

    def test_get_dataset_revision_wrong_project(
        self,
        fxt_dataset_service: DatasetService,
        fxt_dataset_revision_service: DatasetRevisionService,
        fxt_project_with_subset_items_on_disk: tuple[Project, list[DatasetItemDB]],
    ) -> None:
        """Test getting a dataset revision with wrong project ID raises error."""
        project, _ = fxt_project_with_subset_items_on_disk
        dataset = fxt_dataset_service.get_dm_dataset(
            project.id, project.task, DatasetItemAnnotationStatus.WITH_ANNOTATIONS, SampleMode.TRAINING
        )

        # Save a revision for the project
        revision_id = fxt_dataset_revision_service.save_revision(
            project_id=project.id,
            dataset=dataset,
        )

        # Try to get the revision with a different project ID
        wrong_project_id = uuid4()
        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_dataset_revision_service.get_dataset_revision(project_id=wrong_project_id, revision_id=revision_id)

        assert excinfo.value.resource_type == ResourceType.DATASET_REVISION
        assert excinfo.value.resource_id == str(revision_id)

    def test_rename_dataset_revision(
        self,
        fxt_dataset_service: DatasetService,
        fxt_dataset_revision_service: DatasetRevisionService,
        fxt_project_with_subset_items_on_disk: tuple[Project, list[DatasetItemDB]],
    ) -> None:
        """Test updating name of a dataset revision"""
        project, _ = fxt_project_with_subset_items_on_disk
        dataset = fxt_dataset_service.get_dm_dataset(
            project.id, project.task, DatasetItemAnnotationStatus.WITH_ANNOTATIONS, SampleMode.TRAINING
        )

        # Save a revision
        revision_id = fxt_dataset_revision_service.save_revision(
            project_id=project.id,
            dataset=dataset,
        )

        new_dr_name = "This is a new dataset revision name"

        # Get the dataset revision before renaming, rename it and get it after
        dr_before_renaming = fxt_dataset_revision_service.get_dataset_revision(
            project_id=project.id, revision_id=revision_id
        )
        name_before_renaming = dr_before_renaming.name
        dr_from_renaming = fxt_dataset_revision_service.rename_dataset_revision(
            project_id=project.id, dataset_revision=dr_before_renaming, new_name=new_dr_name
        )
        dr_after_renaming = fxt_dataset_revision_service.get_dataset_revision(
            project_id=project.id, revision_id=revision_id
        )

        assert dr_from_renaming.name == new_dr_name
        assert name_before_renaming != new_dr_name
        assert dr_after_renaming.name == new_dr_name

    def test_count_dataset_revision_items(
        self,
        fxt_dataset_service: DatasetService,
        fxt_dataset_revision_service: DatasetRevisionService,
        fxt_project_with_subset_items_on_disk: tuple[Project, list[tuple[MediaDB, DatasetItemDB]]],
    ) -> None:
        """Test counting dataset items by subset."""
        # Create non-empty dataset in memory
        project, media_and_dataset_items = fxt_project_with_subset_items_on_disk
        dataset = fxt_dataset_service.get_dm_dataset(
            project.id,
            project.task,
            annotation_status=DatasetItemAnnotationStatus.WITH_ANNOTATIONS,
            sample_mode=SampleMode.TRAINING,
        )

        # Count items in each subset
        counts = fxt_dataset_revision_service._count_dataset_revision_items(
            dataset=dataset,
        )

        # Calculate expected counts from fixture data
        expected_counts: dict[str, int] = {}
        for _, dataset_item in media_and_dataset_items:
            subset_name = dataset_item.subset if dataset_item.subset is not None else DatasetItemSubset.UNASSIGNED.name
            expected_counts[subset_name] = expected_counts.get(subset_name, 0) + 1
        expected_total = sum(expected_counts.values())

        # Verify counts match expected values from fixture
        assert counts is not None
        assert counts.training == expected_counts["training"]
        assert counts.validation == expected_counts["validation"]
        assert counts.testing == expected_counts["testing"]
        assert counts.total == expected_total

    def test_delete_dataset_revision_files(
        self,
        fxt_projects_dir: Path,
        fxt_dataset_service: DatasetService,
        fxt_dataset_revision_service: DatasetRevisionService,
        fxt_project_with_subset_items_on_disk: tuple[Project, list[DatasetItemDB]],
    ) -> None:
        """Test deleting dataset revision files."""
        project, _ = fxt_project_with_subset_items_on_disk
        dataset = fxt_dataset_service.get_dm_dataset(
            project.id, project.task, DatasetItemAnnotationStatus.WITH_ANNOTATIONS, SampleMode.TRAINING
        )

        # Save a revision
        revision_id = fxt_dataset_revision_service.save_revision(
            project_id=project.id,
            dataset=dataset,
        )

        revision_path = fxt_projects_dir / str(project.id) / "dataset_revisions" / str(revision_id)
        assert revision_path.exists()
        assert (revision_path / "data.parquet").exists()

        # Delete the revision files
        fxt_dataset_revision_service.delete_dataset_revision_files(project_id=project.id, revision_id=revision_id)

        # Verify the files were deleted
        assert not revision_path.exists()

        # Verify the database record is marked as deleted
        revision = fxt_dataset_revision_service.get_dataset_revision(project_id=project.id, revision_id=revision_id)
        assert revision.files_deleted is True

    def test_delete_dataset_revision(
        self,
        fxt_projects_dir: Path,
        fxt_dataset_service: DatasetService,
        fxt_dataset_revision_service: DatasetRevisionService,
        fxt_project_with_subset_items_on_disk: tuple[Project, list[DatasetItemDB]],
    ) -> None:
        """Test deleting dataset revision files."""
        project, _ = fxt_project_with_subset_items_on_disk
        dataset = fxt_dataset_service.get_dm_dataset(
            project.id, project.task, DatasetItemAnnotationStatus.WITH_ANNOTATIONS, SampleMode.TRAINING
        )

        # Save a revision
        revision_id = fxt_dataset_revision_service.save_revision(
            project_id=project.id,
            dataset=dataset,
        )

        revision_path = fxt_projects_dir / str(project.id) / "dataset_revisions" / str(revision_id)
        assert revision_path.exists()
        assert (revision_path / "data.parquet").exists()

        # Delete the revision files
        fxt_dataset_revision_service.delete_dataset_revision(project_id=project.id, revision_id=revision_id)

        # Verify the files were deleted
        assert not revision_path.exists()

        # Verify the database record is removed
        with pytest.raises(ResourceNotFoundError):
            fxt_dataset_revision_service.get_dataset_revision(project_id=project.id, revision_id=revision_id)

    def test_delete_dataset_revision_files_not_found(
        self,
        fxt_dataset_revision_service: DatasetRevisionService,
        fxt_project_with_subset_items: tuple[Project, list[DatasetItemDB]],
    ) -> None:
        """Test deleting files for a non-existent dataset revision raises error."""
        project, _ = fxt_project_with_subset_items
        non_existent_id = uuid4()

        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_dataset_revision_service.delete_dataset_revision_files(
                project_id=project.id, revision_id=non_existent_id
            )

        assert excinfo.value.resource_type == ResourceType.DATASET_REVISION
        assert excinfo.value.resource_id == str(non_existent_id)

    def test_delete_dataset_revision_files_already_deleted(
        self,
        fxt_projects_dir: Path,
        fxt_dataset_service: DatasetService,
        fxt_dataset_revision_service: DatasetRevisionService,
        fxt_project_with_subset_items_on_disk: tuple[Project, list[DatasetItemDB]],
        db_session: Session,
    ) -> None:
        """Test deleting dataset revision files that are already deleted is idempotent."""
        project, _ = fxt_project_with_subset_items_on_disk
        dataset = fxt_dataset_service.get_dm_dataset(
            project.id, project.task, DatasetItemAnnotationStatus.WITH_ANNOTATIONS, SampleMode.TRAINING
        )

        # Save a revision
        revision_id = fxt_dataset_revision_service.save_revision(
            project_id=project.id,
            dataset=dataset,
        )

        # Delete the revision files once
        fxt_dataset_revision_service.delete_dataset_revision_files(project_id=project.id, revision_id=revision_id)

        # Verify files are deleted
        revision_path = fxt_projects_dir / str(project.id) / "dataset_revisions" / str(revision_id)
        assert not revision_path.exists()

        # Delete again - should not raise an error
        fxt_dataset_revision_service.delete_dataset_revision_files(project_id=project.id, revision_id=revision_id)

        # Verify it's still marked as deleted
        revision = fxt_dataset_revision_service.get_dataset_revision(project_id=project.id, revision_id=revision_id)
        assert revision.files_deleted is True

    def test_delete_dataset_revision_files_no_directory(
        self,
        fxt_projects_dir: Path,
        fxt_dataset_revision_service: DatasetRevisionService,
        fxt_project_with_subset_items: tuple[Project, list[DatasetItemDB]],
        db_session: Session,
    ) -> None:
        """Test deleting dataset revision files when directory doesn't exist."""
        project, _ = fxt_project_with_subset_items

        # Create a revision record in the database without creating files
        revision_id = uuid4()
        db_revision = DatasetRevisionDB(
            id=str(revision_id),
            project_id=str(project.id),
            name=f"Dataset ({str(revision_id).split('-')[0]})",
            files_deleted=False,
        )
        db_session.add(db_revision)
        db_session.flush()

        revision_path = fxt_projects_dir / str(project.id) / "dataset_revisions" / str(revision_id)
        assert not revision_path.exists()

        # Delete the revision files - should not raise an error even though directory doesn't exist
        fxt_dataset_revision_service.delete_dataset_revision_files(project_id=project.id, revision_id=revision_id)

        # Verify it's marked as deleted
        revision = fxt_dataset_revision_service.get_dataset_revision(project_id=project.id, revision_id=revision_id)
        assert revision.files_deleted is True

    def test_load_revision(
        self,
        fxt_dataset_service: DatasetService,
        fxt_dataset_revision_service: DatasetRevisionService,
        fxt_project_with_subset_items_on_disk: tuple[Project, list[DatasetItemDB]],
    ) -> None:
        """Test loading a dataset revision as a Datumaro dataset."""
        project, _ = fxt_project_with_subset_items_on_disk
        dataset = fxt_dataset_service.get_dm_dataset(
            project.id, project.task, DatasetItemAnnotationStatus.WITH_ANNOTATIONS, SampleMode.TRAINING
        )

        # Save a revision
        revision_id = fxt_dataset_revision_service.save_revision(
            project_id=project.id,
            dataset=dataset,
        )

        # Load the revision
        loaded_dataset = fxt_dataset_revision_service.load_revision(
            project_id=project.id, dataset_revision_id=revision_id
        )

        # Verify it returns a Datumaro dataset
        assert isinstance(loaded_dataset, dm.Dataset)

    def test_load_revision_files_deleted(
        self,
        fxt_dataset_service: DatasetService,
        fxt_dataset_revision_service: DatasetRevisionService,
        fxt_project_with_subset_items_on_disk: tuple[Project, list[DatasetItemDB]],
    ) -> None:
        """Test loading a revision with deleted files raises error."""
        project, _ = fxt_project_with_subset_items_on_disk
        dataset = fxt_dataset_service.get_dm_dataset(
            project.id, project.task, DatasetItemAnnotationStatus.WITH_ANNOTATIONS, SampleMode.TRAINING
        )

        # Save a revision
        revision_id = fxt_dataset_revision_service.save_revision(
            project_id=project.id,
            dataset=dataset,
        )

        # Delete the revision files
        fxt_dataset_revision_service.delete_dataset_revision_files(project_id=project.id, revision_id=revision_id)

        # Try to load the revision - should raise error
        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_dataset_revision_service.load_revision(project_id=project.id, dataset_revision_id=revision_id)

        assert excinfo.value.resource_type == ResourceType.DATASET_REVISION
        assert excinfo.value.resource_id == str(revision_id)

    def test_get_dataset_revision_item(
        self,
        fxt_dataset_revision_service: DatasetRevisionService,
        fxt_dataset_revision_with_parquet: tuple[Project, UUID],
    ) -> None:
        """Test getting a specific dataset revision item."""
        project, revision_id = fxt_dataset_revision_with_parquet

        # Get the revision
        revision = fxt_dataset_revision_service.get_dataset_revision(project_id=project.id, revision_id=revision_id)

        # Read the parquet to get the item_id
        revision_path = (
            fxt_dataset_revision_service.projects_dir / str(project.id) / "dataset_revisions" / str(revision_id)
        )
        df = pl.read_parquet(revision_path / "data.parquet")
        item_id = df["id"][0]

        # Get the item
        item = fxt_dataset_revision_service.get_dataset_revision_item(
            project_id=project.id,
            dataset_revision=revision,
            item_id=item_id,
        )

        assert item.id == UUID(item_id)
        assert item.format == "jpg"
        assert item.width == 1024
        assert item.height == 768
        assert item.subset == DatasetItemSubset.TRAINING
        assert item.image_path.exists()

    def test_get_dataset_revision_item_not_found(
        self,
        fxt_dataset_revision_service: DatasetRevisionService,
        fxt_dataset_revision_with_parquet: tuple[Project, UUID],
    ) -> None:
        """Test getting a non-existent dataset revision item raises error."""
        project, revision_id = fxt_dataset_revision_with_parquet

        revision = fxt_dataset_revision_service.get_dataset_revision(project_id=project.id, revision_id=revision_id)

        non_existent_id = str(uuid4())

        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_dataset_revision_service.get_dataset_revision_item(
                project_id=project.id,
                dataset_revision=revision,
                item_id=non_existent_id,
            )

        assert excinfo.value.resource_type == ResourceType.DATASET_ITEM
        assert excinfo.value.resource_id == non_existent_id

    def test_get_dataset_revision_item_thumbnail(
        self,
        fxt_dataset_revision_service: DatasetRevisionService,
        fxt_dataset_revision_with_parquet: tuple[Project, UUID],
    ) -> None:
        """Test generating a thumbnail for a dataset revision item."""
        project, revision_id = fxt_dataset_revision_with_parquet

        revision = fxt_dataset_revision_service.get_dataset_revision(project_id=project.id, revision_id=revision_id)

        # Read the parquet to get the item_id
        revision_path = (
            fxt_dataset_revision_service.projects_dir / str(project.id) / "dataset_revisions" / str(revision_id)
        )
        df = pl.read_parquet(revision_path / "data.parquet")
        item_id = df["id"][0]

        # Get the thumbnail
        thumbnail = fxt_dataset_revision_service.get_dataset_revision_item_thumbnail(
            project_id=project.id,
            dataset_revision=revision,
            item_id=item_id,
        )

        assert isinstance(thumbnail, Image.Image)
        assert thumbnail.width == thumbnail.height == DATASET_REVISION_ITEM_THUMBNAIL_SIZE
        # The thumbnail must be in a mode that can be saved as JPEG downstream
        # (write_image_to_response()); otherwise saving would fail at request time.
        assert thumbnail.mode in ("RGB", "L", "CMYK")

    def test_get_dataset_revision_item_thumbnail_high_bit_depth_is_normalized(
        self,
        fxt_dataset_revision_service: DatasetRevisionService,
        fxt_dataset_revision_with_16bit_parquet: tuple[Project, UUID],
    ) -> None:
        """Thumbnails for 16-bit images are normalized and JPEG-saveable.

        Without normalization, a naive .convert("RGB") keeps only the high byte of a
        16-bit image. For values confined to a narrow high band (60000-65000) this yields a
        washed-out, near-white thumbnail, and a high-bit-depth mode (e.g. I;16) would only
        fail later when write_image_to_response() tries to save it as JPEG.
        """
        project, revision_id = fxt_dataset_revision_with_16bit_parquet

        revision = fxt_dataset_revision_service.get_dataset_revision(project_id=project.id, revision_id=revision_id)

        revision_path = (
            fxt_dataset_revision_service.projects_dir / str(project.id) / "dataset_revisions" / str(revision_id)
        )
        df = pl.read_parquet(revision_path / "data.parquet")
        item_id = df["id"][0]

        thumbnail = fxt_dataset_revision_service.get_dataset_revision_item_thumbnail(
            project_id=project.id,
            dataset_revision=revision,
            item_id=item_id,
        )

        assert isinstance(thumbnail, Image.Image)
        assert thumbnail.width == thumbnail.height == DATASET_REVISION_ITEM_THUMBNAIL_SIZE

        # 1. The thumbnail must be in a JPEG-compatible mode, not a high-bit-depth one.
        assert thumbnail.mode in ("RGB", "L", "CMYK")
        assert thumbnail.mode not in ("I;16", "I;16B", "I;16L", "I;16S", "I;16BS", "I", "F")

        # 2. Normalization must spread the narrow input band across the full [0, 255] range
        thumbnail_arr = np.array(thumbnail.convert("L"))
        assert thumbnail_arr.min() < 32, "Dark end of the value range should be preserved after normalization"
        assert thumbnail_arr.max() > 223, "Bright end of the value range should be preserved after normalization"

        # The thumbnail must be savable as JPEG without raising errors (mirrors write_image_to_response()).
        from io import BytesIO

        buffer = BytesIO()
        thumbnail.save(buffer, format="JPEG")

    def test_get_dataset_revision_item_thumbnail_not_found(
        self,
        fxt_dataset_revision_service: DatasetRevisionService,
        fxt_dataset_revision_with_parquet: tuple[Project, UUID],
    ) -> None:
        """Test getting thumbnail for non-existent item raises error."""
        project, revision_id = fxt_dataset_revision_with_parquet

        revision = fxt_dataset_revision_service.get_dataset_revision(project_id=project.id, revision_id=revision_id)

        non_existent_id = str(uuid4())

        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_dataset_revision_service.get_dataset_revision_item_thumbnail(
                project_id=project.id,
                dataset_revision=revision,
                item_id=non_existent_id,
            )

        assert excinfo.value.resource_type == ResourceType.DATASET_ITEM
        assert excinfo.value.resource_id == non_existent_id


class TestDatasetRevisionServiceDatasetViews:
    """Tests for dataset revisions created from a dataset view."""

    @staticmethod
    def _save_revision(
        dataset_service: DatasetService,
        dataset_revision_service: DatasetRevisionService,
        project: Project,
        dataset_view_id: UUID | None = None,
        dataset_view_name: str | None = None,
    ) -> UUID:
        dataset = dataset_service.get_dm_dataset(
            project.id,
            project.task,
            DatasetItemAnnotationStatus.WITH_ANNOTATIONS,
            SampleMode.TRAINING,
            dataset_view_id=dataset_view_id,
        )
        return dataset_revision_service.save_revision(
            project_id=project.id,
            dataset=dataset,
            dataset_view_id=dataset_view_id,
            dataset_view_name=dataset_view_name,
        )

    def test_save_revision_from_view_records_scope_and_name(
        self,
        fxt_dataset_service: DatasetService,
        fxt_dataset_revision_service: DatasetRevisionService,
        fxt_dataset_view_service: DatasetViewService,
        fxt_project_with_subset_items_on_disk: tuple[Project, list[tuple[MediaDB, DatasetItemDB]]],
    ) -> None:
        project, media_and_dataset_items = fxt_project_with_subset_items_on_disk
        view = fxt_dataset_view_service.create_dataset_view(
            project_id=project.id,
            name="Collection One",
            media_ids=[UUID(media.id) for media, _ in media_and_dataset_items],
        )

        revision_id = self._save_revision(
            fxt_dataset_service, fxt_dataset_revision_service, project, view.id, view.name
        )

        revision = fxt_dataset_revision_service.get_dataset_revision(project_id=project.id, revision_id=revision_id)
        assert revision.dataset_view_id == view.id
        assert revision.name == f"Collection One ({str(revision_id).split('-')[0]})"

    def test_uptodate_revision_is_reused_for_the_same_view(
        self,
        fxt_dataset_service: DatasetService,
        fxt_dataset_revision_service: DatasetRevisionService,
        fxt_dataset_view_service: DatasetViewService,
        fxt_project_with_subset_items_on_disk: tuple[Project, list[tuple[MediaDB, DatasetItemDB]]],
    ) -> None:
        project, media_and_dataset_items = fxt_project_with_subset_items_on_disk
        view = fxt_dataset_view_service.create_dataset_view(
            project_id=project.id,
            name="Collection One",
            media_ids=[UUID(media.id) for media, _ in media_and_dataset_items],
        )
        revision_id = self._save_revision(
            fxt_dataset_service, fxt_dataset_revision_service, project, view.id, view.name
        )

        latest = fxt_dataset_revision_service.get_latest_uptodate_dataset_revision(
            project_id=project.id, dataset_view_id=view.id
        )

        assert latest is not None
        assert latest.id == revision_id

    def test_view_revision_is_not_reused_for_the_entire_dataset(
        self,
        fxt_dataset_service: DatasetService,
        fxt_dataset_revision_service: DatasetRevisionService,
        fxt_dataset_view_service: DatasetViewService,
        fxt_project_with_subset_items_on_disk: tuple[Project, list[tuple[MediaDB, DatasetItemDB]]],
    ) -> None:
        project, media_and_dataset_items = fxt_project_with_subset_items_on_disk
        view = fxt_dataset_view_service.create_dataset_view(
            project_id=project.id,
            name="Collection One",
            media_ids=[UUID(media.id) for media, _ in media_and_dataset_items],
        )
        self._save_revision(fxt_dataset_service, fxt_dataset_revision_service, project, view.id, view.name)

        latest = fxt_dataset_revision_service.get_latest_uptodate_dataset_revision(project_id=project.id)

        assert latest is None

    def test_full_dataset_revision_is_not_reused_for_a_view(
        self,
        fxt_dataset_service: DatasetService,
        fxt_dataset_revision_service: DatasetRevisionService,
        fxt_dataset_view_service: DatasetViewService,
        fxt_project_with_subset_items_on_disk: tuple[Project, list[tuple[MediaDB, DatasetItemDB]]],
    ) -> None:
        project, media_and_dataset_items = fxt_project_with_subset_items_on_disk
        self._save_revision(fxt_dataset_service, fxt_dataset_revision_service, project)
        view = fxt_dataset_view_service.create_dataset_view(
            project_id=project.id,
            name="Collection One",
            media_ids=[UUID(media.id) for media, _ in media_and_dataset_items],
        )

        latest = fxt_dataset_revision_service.get_latest_uptodate_dataset_revision(
            project_id=project.id, dataset_view_id=view.id
        )

        assert latest is None

    def test_view_revision_is_invalidated_by_membership_change(
        self,
        fxt_dataset_service: DatasetService,
        fxt_dataset_revision_service: DatasetRevisionService,
        fxt_dataset_view_service: DatasetViewService,
        fxt_project_with_subset_items_on_disk: tuple[Project, list[tuple[MediaDB, DatasetItemDB]]],
    ) -> None:
        project, media_and_dataset_items = fxt_project_with_subset_items_on_disk
        media_ids = [UUID(media.id) for media, _ in media_and_dataset_items]
        view = fxt_dataset_view_service.create_dataset_view(
            project_id=project.id, name="Collection One", media_ids=media_ids
        )
        self._save_revision(fxt_dataset_service, fxt_dataset_revision_service, project, view.id, view.name)

        # Removing a medium from the view changes the data the revision covers, so it must no longer be reused.
        fxt_dataset_view_service.unassign_media(
            project_id=project.id, dataset_view_id=view.id, media_ids=[media_ids[0]]
        )

        latest = fxt_dataset_revision_service.get_latest_uptodate_dataset_revision(
            project_id=project.id, dataset_view_id=view.id
        )

        assert latest is None

    def test_view_revision_is_invalidated_by_annotation_change(
        self,
        fxt_dataset_service: DatasetService,
        fxt_dataset_revision_service: DatasetRevisionService,
        fxt_dataset_view_service: DatasetViewService,
        fxt_project_with_subset_items_on_disk: tuple[Project, list[tuple[MediaDB, DatasetItemDB]]],
    ) -> None:
        project, media_and_dataset_items = fxt_project_with_subset_items_on_disk
        view = fxt_dataset_view_service.create_dataset_view(
            project_id=project.id,
            name="Collection One",
            media_ids=[UUID(media.id) for media, _ in media_and_dataset_items],
        )
        self._save_revision(fxt_dataset_service, fxt_dataset_revision_service, project, view.id, view.name)

        unassigned_item = next(
            ds_item
            for _, ds_item in media_and_dataset_items
            if ds_item.subset == DatasetItemSubset.UNASSIGNED.name.lower()
        )
        fxt_dataset_service.assign_dataset_item_subset(
            project_id=project.id, dataset_item_id=UUID(unassigned_item.id), subset=DatasetItemSubset.TRAINING
        )

        latest = fxt_dataset_revision_service.get_latest_uptodate_dataset_revision(
            project_id=project.id, dataset_view_id=view.id
        )

        assert latest is None

    def test_view_revision_survives_changes_outside_the_view(
        self,
        fxt_dataset_service: DatasetService,
        fxt_dataset_revision_service: DatasetRevisionService,
        fxt_dataset_view_service: DatasetViewService,
        fxt_project_with_subset_items_on_disk: tuple[Project, list[tuple[MediaDB, DatasetItemDB]]],
    ) -> None:
        project, media_and_dataset_items = fxt_project_with_subset_items_on_disk
        unassigned_item = next(
            ds_item
            for _, ds_item in media_and_dataset_items
            if ds_item.subset == DatasetItemSubset.UNASSIGNED.name.lower()
        )
        # The view deliberately excludes the item that is modified afterwards.
        view = fxt_dataset_view_service.create_dataset_view(
            project_id=project.id,
            name="Collection One",
            media_ids=[UUID(media.id) for media, item in media_and_dataset_items if item.id != unassigned_item.id],
        )
        revision_id = self._save_revision(
            fxt_dataset_service, fxt_dataset_revision_service, project, view.id, view.name
        )

        fxt_dataset_service.assign_dataset_item_subset(
            project_id=project.id, dataset_item_id=UUID(unassigned_item.id), subset=DatasetItemSubset.TRAINING
        )

        latest = fxt_dataset_revision_service.get_latest_uptodate_dataset_revision(
            project_id=project.id, dataset_view_id=view.id
        )

        assert latest is not None
        assert latest.id == revision_id

    def test_view_revision_is_invalidated_by_media_deletion(
        self,
        fxt_dataset_service: DatasetService,
        fxt_dataset_revision_service: DatasetRevisionService,
        fxt_dataset_view_service: DatasetViewService,
        fxt_media_service: MediaService,
        fxt_project_with_subset_items_on_disk: tuple[Project, list[tuple[MediaDB, DatasetItemDB]]],
    ) -> None:
        project, media_and_dataset_items = fxt_project_with_subset_items_on_disk
        media_ids = [UUID(media.id) for media, _ in media_and_dataset_items]
        view = fxt_dataset_view_service.create_dataset_view(
            project_id=project.id, name="Collection One", media_ids=media_ids
        )
        self._save_revision(fxt_dataset_service, fxt_dataset_revision_service, project, view.id, view.name)

        # Deleting a medium cascades its view assignment, so the view must still be marked as changed.
        fxt_media_service.delete_media(project=project, media_id=media_ids[0])

        latest = fxt_dataset_revision_service.get_latest_uptodate_dataset_revision(
            project_id=project.id, dataset_view_id=view.id
        )

        assert latest is None

    def test_view_revision_is_not_reused_after_the_view_is_deleted(
        self,
        fxt_dataset_service: DatasetService,
        fxt_dataset_revision_service: DatasetRevisionService,
        fxt_dataset_view_service: DatasetViewService,
        fxt_project_with_subset_items_on_disk: tuple[Project, list[tuple[MediaDB, DatasetItemDB]]],
    ) -> None:
        project, media_and_dataset_items = fxt_project_with_subset_items_on_disk
        view = fxt_dataset_view_service.create_dataset_view(
            project_id=project.id,
            name="Collection One",
            media_ids=[UUID(media.id) for media, _ in media_and_dataset_items],
        )
        self._save_revision(fxt_dataset_service, fxt_dataset_revision_service, project, view.id, view.name)

        fxt_dataset_view_service.delete_dataset_view(project_id=project.id, dataset_view_id=view.id)

        # The revision keeps its scope marker, so it is neither reusable as a full-dataset one nor reachable.
        assert fxt_dataset_revision_service.get_latest_uptodate_dataset_revision(project_id=project.id) is None
        assert (
            fxt_dataset_revision_service.get_latest_uptodate_dataset_revision(
                project_id=project.id, dataset_view_id=view.id
            )
            is None
        )

    def test_save_revision_from_view_without_all_subsets_raises(
        self,
        fxt_dataset_service: DatasetService,
        fxt_dataset_revision_service: DatasetRevisionService,
        fxt_dataset_view_service: DatasetViewService,
        fxt_project_with_subset_items_on_disk: tuple[Project, list[tuple[MediaDB, DatasetItemDB]]],
    ) -> None:
        project, media_and_dataset_items = fxt_project_with_subset_items_on_disk
        training_media = [
            UUID(media.id)
            for media, item in media_and_dataset_items
            if item.subset == DatasetItemSubset.TRAINING.name.lower()
        ]
        view = fxt_dataset_view_service.create_dataset_view(
            project_id=project.id, name="Only training", media_ids=training_media
        )

        with pytest.raises(ValueError, match="Only training"):
            self._save_revision(fxt_dataset_service, fxt_dataset_revision_service, project, view.id, view.name)
