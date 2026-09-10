# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from datetime import datetime
from uuid import UUID, uuid4

import pytest
from sqlalchemy.orm import Session

from app.db.schema import DatasetItemDB, DatasetItemLabelDB, DatasetViewItemDB, MediaDB, PipelineDB
from app.models import DatasetItemAnnotationStatus, DatasetItemSubset, Pipeline, Project, Video
from app.services.base import ResourceNotFoundError, ResourceWithNameAlreadyExistsError
from app.services.dataset_service import DatasetItemFilters
from app.services.dataset_view_service import DatasetViewService
from app.services.media_service import MediaFilters


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
def fxt_media_factory(db_session: Session) -> Callable[[str, list[dict]], list[MediaDB]]:
    """Returns a callable that creates and persists MediaDB objects for a project."""

    def _create_media(project_id: str, configs: list[dict]) -> list[MediaDB]:
        items = []
        for config in configs:
            m = MediaDB(**config)
            m.project_id = project_id
            m.created_at = datetime.fromisoformat("2025-02-01T00:00:00+00:00")
            items.append(m)
        db_session.add_all(items)
        db_session.flush()
        return items

    return _create_media


@pytest.fixture
def fxt_dataset_item_factory(db_session: Session) -> Callable[[str, str, dict], DatasetItemDB]:
    """Returns a callable that creates and persists a DatasetItemDB for a given media id."""

    def _create_dataset_item(project_id: str, media_id: str, config: dict) -> DatasetItemDB:
        dataset_item = DatasetItemDB(id=media_id, project_id=project_id, **config)
        dataset_item.created_at = datetime.fromisoformat("2025-02-01T00:00:00+00:00")
        db_session.add(dataset_item)
        db_session.flush()
        return dataset_item

    return _create_dataset_item


@pytest.fixture
def fxt_project_with_media(
    fxt_project_with_pipeline, fxt_media_factory, fxt_dataset_item_factory, db_session
) -> tuple[Project, dict[str, MediaDB]]:
    """
    Sets up a project with:
      - image1: assigned to a dataset item without annotations
      - image2: assigned to a dataset item with annotations (labels[0])
      - video1: with 2 frames, one annotated (labels[1]) and one not
    """
    project, _ = fxt_project_with_pipeline
    label_0, label_1 = project.task.labels[0], project.task.labels[1]

    image1, image2 = fxt_media_factory(
        str(project.id),
        [
            {"type": "image", "name": "image1", "format": "jpg", "size": 1024, "width": 1024, "height": 768},
            {"type": "image", "name": "image2", "format": "jpg", "size": 1024, "width": 1024, "height": 768},
        ],
    )
    fxt_dataset_item_factory(str(project.id), image1.id, {"subset": "unassigned"})
    fxt_dataset_item_factory(
        str(project.id),
        image2.id,
        {
            "subset": "training",
            "user_reviewed": True,
            "annotation_data": [{"labels": [{"id": str(label_0.id)}], "shape": {"type": "full_image"}}],
        },
    )
    db_session.add(DatasetItemLabelDB(dataset_item_id=image2.id, label_id=str(label_0.id)))

    (video1,) = fxt_media_factory(
        str(project.id),
        [
            {
                "type": "video",
                "name": "video1",
                "format": "avi",
                "size": 2048,
                "width": 640,
                "height": 480,
                "fps": 25.0,
                "frame_count": 100,
            }
        ],
    )
    frame_annotated, frame_unannotated = fxt_media_factory(
        str(project.id),
        [
            {
                "type": "video_frame",
                "name": "video1_10",
                "format": "jpg",
                "size": 512,
                "width": 640,
                "height": 480,
                "video_id": video1.id,
                "frame_index": 10,
            },
            {
                "type": "video_frame",
                "name": "video1_20",
                "format": "jpg",
                "size": 512,
                "width": 640,
                "height": 480,
                "video_id": video1.id,
                "frame_index": 20,
            },
        ],
    )
    fxt_dataset_item_factory(
        str(project.id),
        frame_annotated.id,
        {
            "subset": "unassigned",
            "user_reviewed": True,
            "annotation_data": [{"labels": [{"id": str(label_1.id)}], "shape": {"type": "full_image"}}],
        },
    )
    db_session.add(DatasetItemLabelDB(dataset_item_id=frame_annotated.id, label_id=str(label_1.id)))
    db_session.flush()

    return project, {
        "image1": image1,
        "image2": image2,
        "video1": video1,
        "frame_annotated": frame_annotated,
        "frame_unannotated": frame_unannotated,
    }


class TestDatasetViewServiceCRUD:
    def test_create_dataset_view_empty(self, fxt_dataset_view_service: DatasetViewService, fxt_project_with_pipeline):
        project, _ = fxt_project_with_pipeline

        view = fxt_dataset_view_service.create_dataset_view(project_id=project.id, name="My view")

        assert view.name == "My view"
        assert view.project_id == project.id

    def test_create_dataset_view_with_media(self, fxt_dataset_view_service: DatasetViewService, fxt_project_with_media):
        project, media = fxt_project_with_media

        view = fxt_dataset_view_service.create_dataset_view(
            project_id=project.id, name="My view", media_ids=[UUID(media["image1"].id), UUID(media["video1"].id)]
        )

        assigned = fxt_dataset_view_service.list_dataset_view_media(project_id=project.id, dataset_view_id=view.id)
        assert {str(m.id) for m in assigned} == {media["image1"].id, media["video1"].id}

    def test_create_dataset_view_with_unknown_media_raises(
        self, fxt_dataset_view_service: DatasetViewService, fxt_project_with_pipeline
    ):
        project, _ = fxt_project_with_pipeline

        with pytest.raises(ResourceNotFoundError):
            fxt_dataset_view_service.create_dataset_view(project_id=project.id, name="My view", media_ids=[uuid4()])

    def test_create_dataset_view_with_video_frame_media_raises(
        self, fxt_dataset_view_service: DatasetViewService, fxt_project_with_media
    ):
        # Dataset views only track membership at the image/video level; a video frame is considered
        # part of a view solely via its parent video, so passing a frame id directly must be rejected.
        project, media = fxt_project_with_media

        with pytest.raises(ResourceNotFoundError):
            fxt_dataset_view_service.create_dataset_view(
                project_id=project.id, name="My view", media_ids=[UUID(media["frame_annotated"].id)]
            )

    def test_create_dataset_view_duplicate_name_raises(
        self, fxt_dataset_view_service: DatasetViewService, fxt_project_with_pipeline
    ):
        project, _ = fxt_project_with_pipeline
        fxt_dataset_view_service.create_dataset_view(project_id=project.id, name="My view")

        with pytest.raises(ResourceWithNameAlreadyExistsError):
            fxt_dataset_view_service.create_dataset_view(project_id=project.id, name="My view")

    def test_list_dataset_views(self, fxt_dataset_view_service: DatasetViewService, fxt_project_with_pipeline):
        project, _ = fxt_project_with_pipeline
        fxt_dataset_view_service.create_dataset_view(project_id=project.id, name="View A")
        fxt_dataset_view_service.create_dataset_view(project_id=project.id, name="View B")

        views = fxt_dataset_view_service.list_dataset_views(project_id=project.id)

        assert {v.name for v in views} == {"View A", "View B"}

    def test_get_dataset_view_by_id_not_found(
        self, fxt_dataset_view_service: DatasetViewService, fxt_project_with_pipeline
    ):
        project, _ = fxt_project_with_pipeline

        with pytest.raises(ResourceNotFoundError):
            fxt_dataset_view_service.get_dataset_view_by_id(project_id=project.id, dataset_view_id=uuid4())

    def test_rename_dataset_view(self, fxt_dataset_view_service: DatasetViewService, fxt_project_with_pipeline):
        project, _ = fxt_project_with_pipeline
        view = fxt_dataset_view_service.create_dataset_view(project_id=project.id, name="Old name")

        renamed = fxt_dataset_view_service.rename_dataset_view(
            project_id=project.id, dataset_view_id=view.id, new_name="New name"
        )

        assert renamed.name == "New name"
        fetched = fxt_dataset_view_service.get_dataset_view_by_id(project_id=project.id, dataset_view_id=view.id)
        assert fetched.name == "New name"

    def test_rename_dataset_view_same_name_is_noop(
        self, fxt_dataset_view_service: DatasetViewService, fxt_project_with_pipeline
    ):
        project, _ = fxt_project_with_pipeline
        view = fxt_dataset_view_service.create_dataset_view(project_id=project.id, name="Same name")

        renamed = fxt_dataset_view_service.rename_dataset_view(
            project_id=project.id, dataset_view_id=view.id, new_name="Same name"
        )

        assert renamed.name == "Same name"

    def test_rename_dataset_view_duplicate_name_raises(
        self, fxt_dataset_view_service: DatasetViewService, fxt_project_with_pipeline
    ):
        project, _ = fxt_project_with_pipeline
        fxt_dataset_view_service.create_dataset_view(project_id=project.id, name="Taken")
        view = fxt_dataset_view_service.create_dataset_view(project_id=project.id, name="Free")

        with pytest.raises(ResourceWithNameAlreadyExistsError):
            fxt_dataset_view_service.rename_dataset_view(
                project_id=project.id, dataset_view_id=view.id, new_name="Taken"
            )

    def test_delete_dataset_view(
        self, fxt_dataset_view_service: DatasetViewService, fxt_project_with_media, db_session: Session
    ):
        project, media = fxt_project_with_media
        view = fxt_dataset_view_service.create_dataset_view(
            project_id=project.id, name="To delete", media_ids=[UUID(media["image1"].id)]
        )

        fxt_dataset_view_service.delete_dataset_view(project_id=project.id, dataset_view_id=view.id)

        with pytest.raises(ResourceNotFoundError):
            fxt_dataset_view_service.get_dataset_view_by_id(project_id=project.id, dataset_view_id=view.id)
        # Cascading delete: the assignment rows are gone too
        remaining = db_session.query(DatasetViewItemDB).filter_by(dataset_view_id=str(view.id)).all()
        assert remaining == []

    def test_delete_dataset_view_not_found(
        self, fxt_dataset_view_service: DatasetViewService, fxt_project_with_pipeline
    ):
        project, _ = fxt_project_with_pipeline

        with pytest.raises(ResourceNotFoundError):
            fxt_dataset_view_service.delete_dataset_view(project_id=project.id, dataset_view_id=uuid4())


class TestDatasetViewServiceAssignment:
    def test_assign_media(self, fxt_dataset_view_service: DatasetViewService, fxt_project_with_media):
        project, media = fxt_project_with_media
        view = fxt_dataset_view_service.create_dataset_view(project_id=project.id, name="My view")

        fxt_dataset_view_service.assign_media(
            project_id=project.id, dataset_view_id=view.id, media_ids=[UUID(media["image1"].id)]
        )

        assigned = fxt_dataset_view_service.list_dataset_view_media(project_id=project.id, dataset_view_id=view.id)
        assert {str(m.id) for m in assigned} == {media["image1"].id}

    def test_assign_media_is_idempotent(self, fxt_dataset_view_service: DatasetViewService, fxt_project_with_media):
        project, media = fxt_project_with_media
        view = fxt_dataset_view_service.create_dataset_view(project_id=project.id, name="My view")

        fxt_dataset_view_service.assign_media(
            project_id=project.id, dataset_view_id=view.id, media_ids=[UUID(media["image1"].id)]
        )
        # Assigning the same media again should be a no-op, not raise
        fxt_dataset_view_service.assign_media(
            project_id=project.id, dataset_view_id=view.id, media_ids=[UUID(media["image1"].id)]
        )

        assert fxt_dataset_view_service.count_dataset_view_media(project_id=project.id, dataset_view_id=view.id) == 1

    def test_assign_unknown_media_raises(self, fxt_dataset_view_service: DatasetViewService, fxt_project_with_media):
        project, _ = fxt_project_with_media
        view = fxt_dataset_view_service.create_dataset_view(project_id=project.id, name="My view")

        with pytest.raises(ResourceNotFoundError):
            fxt_dataset_view_service.assign_media(project_id=project.id, dataset_view_id=view.id, media_ids=[uuid4()])

    def test_assign_video_frame_media_raises(
        self, fxt_dataset_view_service: DatasetViewService, fxt_project_with_media
    ):
        # Video frames cannot be assigned directly to a view; only their parent video can be. Frame
        # membership is derived implicitly once the video itself is assigned (see other tests).
        project, media = fxt_project_with_media
        view = fxt_dataset_view_service.create_dataset_view(project_id=project.id, name="My view")

        with pytest.raises(ResourceNotFoundError):
            fxt_dataset_view_service.assign_media(
                project_id=project.id, dataset_view_id=view.id, media_ids=[UUID(media["frame_unannotated"].id)]
            )

    def test_assign_media_dataset_view_not_found(
        self, fxt_dataset_view_service: DatasetViewService, fxt_project_with_media
    ):
        project, media = fxt_project_with_media

        with pytest.raises(ResourceNotFoundError):
            fxt_dataset_view_service.assign_media(
                project_id=project.id, dataset_view_id=uuid4(), media_ids=[UUID(media["image1"].id)]
            )

    def test_unassign_media(self, fxt_dataset_view_service: DatasetViewService, fxt_project_with_media):
        project, media = fxt_project_with_media
        view = fxt_dataset_view_service.create_dataset_view(
            project_id=project.id,
            name="My view",
            media_ids=[UUID(media["image1"].id), UUID(media["image2"].id)],
        )

        fxt_dataset_view_service.unassign_media(
            project_id=project.id, dataset_view_id=view.id, media_ids=[UUID(media["image1"].id)]
        )

        assigned = fxt_dataset_view_service.list_dataset_view_media(project_id=project.id, dataset_view_id=view.id)
        assert {str(m.id) for m in assigned} == {media["image2"].id}

    def test_unassign_media_not_previously_assigned_is_noop(
        self, fxt_dataset_view_service: DatasetViewService, fxt_project_with_media
    ):
        project, media = fxt_project_with_media
        view = fxt_dataset_view_service.create_dataset_view(project_id=project.id, name="My view")

        # Should not raise even though the media was never assigned
        fxt_dataset_view_service.unassign_media(
            project_id=project.id, dataset_view_id=view.id, media_ids=[UUID(media["image1"].id)]
        )

    def test_unassign_media_does_not_delete_media(
        self, fxt_dataset_view_service: DatasetViewService, fxt_project_with_media, fxt_media_service
    ):
        project, media = fxt_project_with_media
        view = fxt_dataset_view_service.create_dataset_view(
            project_id=project.id, name="My view", media_ids=[UUID(media["image1"].id)]
        )

        fxt_dataset_view_service.unassign_media(
            project_id=project.id, dataset_view_id=view.id, media_ids=[UUID(media["image1"].id)]
        )

        # Media still exists in the project's main dataset
        still_there = fxt_media_service.get_media_by_id(project_id=project.id, media_id=UUID(media["image1"].id))
        assert still_there is not None

    def test_assign_media_to_multiple_views(self, fxt_dataset_view_service: DatasetViewService, fxt_project_with_media):
        project, media = fxt_project_with_media
        view_a = fxt_dataset_view_service.create_dataset_view(
            project_id=project.id, name="View A", media_ids=[UUID(media["image1"].id)]
        )
        view_b = fxt_dataset_view_service.create_dataset_view(
            project_id=project.id, name="View B", media_ids=[UUID(media["image1"].id)]
        )

        media_a = fxt_dataset_view_service.list_dataset_view_media(project_id=project.id, dataset_view_id=view_a.id)
        media_b = fxt_dataset_view_service.list_dataset_view_media(project_id=project.id, dataset_view_id=view_b.id)
        assert {str(m.id) for m in media_a} == {media["image1"].id}
        assert {str(m.id) for m in media_b} == {media["image1"].id}


class TestDatasetViewServiceMedia:
    def test_list_dataset_view_media_excludes_frames(
        self, fxt_dataset_view_service: DatasetViewService, fxt_project_with_media
    ):
        project, media = fxt_project_with_media
        view = fxt_dataset_view_service.create_dataset_view(
            project_id=project.id, name="My view", media_ids=[UUID(media["video1"].id)]
        )

        assigned = fxt_dataset_view_service.list_dataset_view_media(project_id=project.id, dataset_view_id=view.id)

        assert len(assigned) == 1
        assert isinstance(assigned[0], Video)
        assert str(assigned[0].id) == media["video1"].id
        # The video's annotated frame count is correctly reported even though frames themselves
        # are not directly assigned to the view.
        assert assigned[0].annotated_frame_count == 1

    def test_list_dataset_view_media_ids(self, fxt_dataset_view_service: DatasetViewService, fxt_project_with_media):
        """list_dataset_view_media_ids should return the same (id, type) pairs as list_dataset_view_media."""
        project, media = fxt_project_with_media
        view = fxt_dataset_view_service.create_dataset_view(
            project_id=project.id, name="My view", media_ids=[UUID(media["image1"].id), UUID(media["video1"].id)]
        )

        assigned = fxt_dataset_view_service.list_dataset_view_media(project_id=project.id, dataset_view_id=view.id)
        assigned_ids = fxt_dataset_view_service.list_dataset_view_media_ids(
            project_id=project.id, dataset_view_id=view.id
        )

        assert set(assigned_ids) == {(m.id, m.type) for m in assigned}

    def test_count_and_list_dataset_view_media_with_filters(
        self, fxt_dataset_view_service: DatasetViewService, fxt_project_with_media
    ):
        project, media = fxt_project_with_media
        view = fxt_dataset_view_service.create_dataset_view(
            project_id=project.id,
            name="My view",
            media_ids=[UUID(media["image1"].id), UUID(media["image2"].id), UUID(media["video1"].id)],
        )

        with_annotations = MediaFilters(annotation_status=DatasetItemAnnotationStatus.WITH_ANNOTATIONS)
        count = fxt_dataset_view_service.count_dataset_view_media(
            project_id=project.id, dataset_view_id=view.id, filters=with_annotations
        )
        listed = fxt_dataset_view_service.list_dataset_view_media(
            project_id=project.id, dataset_view_id=view.id, filters=with_annotations
        )

        # image2 has its own annotation, video1 has an annotated frame => both count as "with annotations"
        assert count == 2
        assert {str(m.id) for m in listed} == {media["image2"].id, media["video1"].id}

    def test_list_dataset_view_media_not_assigned_are_excluded(
        self, fxt_dataset_view_service: DatasetViewService, fxt_project_with_media
    ):
        project, media = fxt_project_with_media
        view = fxt_dataset_view_service.create_dataset_view(
            project_id=project.id, name="My view", media_ids=[UUID(media["image1"].id)]
        )

        listed = fxt_dataset_view_service.list_dataset_view_media(project_id=project.id, dataset_view_id=view.id)

        assert {str(m.id) for m in listed} == {media["image1"].id}

    def test_media_operations_dataset_view_not_found(
        self, fxt_dataset_view_service: DatasetViewService, fxt_project_with_media
    ):
        project, _ = fxt_project_with_media

        with pytest.raises(ResourceNotFoundError):
            fxt_dataset_view_service.list_dataset_view_media(project_id=project.id, dataset_view_id=uuid4())
        with pytest.raises(ResourceNotFoundError):
            fxt_dataset_view_service.count_dataset_view_media(project_id=project.id, dataset_view_id=uuid4())


class TestDatasetViewServiceItems:
    def test_list_dataset_view_items_includes_frames_of_assigned_video(
        self, fxt_dataset_view_service: DatasetViewService, fxt_project_with_media
    ):
        project, media = fxt_project_with_media
        view = fxt_dataset_view_service.create_dataset_view(
            project_id=project.id, name="My view", media_ids=[UUID(media["video1"].id)]
        )

        items = fxt_dataset_view_service.list_dataset_view_items(project_id=project.id, dataset_view_id=view.id)

        # Only the annotated frame has a dataset item; it should show up even though the frame itself
        # was never directly assigned to the view (only its parent video was).
        assert {str(item.id) for item in items} == {media["frame_annotated"].id}

    def test_count_and_list_dataset_view_items_with_filters(
        self, fxt_dataset_view_service: DatasetViewService, fxt_project_with_media
    ):
        project, media = fxt_project_with_media
        view = fxt_dataset_view_service.create_dataset_view(
            project_id=project.id,
            name="My view",
            media_ids=[UUID(media["image1"].id), UUID(media["image2"].id), UUID(media["video1"].id)],
        )

        training_only = DatasetItemFilters(subsets=[DatasetItemSubset.TRAINING])
        count = fxt_dataset_view_service.count_dataset_view_items(
            project_id=project.id, dataset_view_id=view.id, filters=training_only
        )
        items = fxt_dataset_view_service.list_dataset_view_items(
            project_id=project.id, dataset_view_id=view.id, filters=training_only
        )

        assert count == 1
        assert {str(item.id) for item in items} == {media["image2"].id}

    def test_list_dataset_view_items_not_assigned_are_excluded(
        self, fxt_dataset_view_service: DatasetViewService, fxt_project_with_media
    ):
        project, media = fxt_project_with_media
        # image2 has an annotated dataset item but is not assigned to the view; only image1 (unannotated) is.
        view = fxt_dataset_view_service.create_dataset_view(
            project_id=project.id, name="My view", media_ids=[UUID(media["image1"].id)]
        )

        items = fxt_dataset_view_service.list_dataset_view_items(project_id=project.id, dataset_view_id=view.id)

        assert {str(item.id) for item in items} == {media["image1"].id}

    def test_item_operations_dataset_view_not_found(
        self, fxt_dataset_view_service: DatasetViewService, fxt_project_with_media
    ):
        project, _ = fxt_project_with_media

        with pytest.raises(ResourceNotFoundError):
            fxt_dataset_view_service.list_dataset_view_items(project_id=project.id, dataset_view_id=uuid4())
        with pytest.raises(ResourceNotFoundError):
            fxt_dataset_view_service.count_dataset_view_items(project_id=project.id, dataset_view_id=uuid4())


class TestDatasetViewServiceStatistics:
    def test_get_dataset_view_statistics(self, fxt_dataset_view_service: DatasetViewService, fxt_project_with_media):
        project, media = fxt_project_with_media
        view = fxt_dataset_view_service.create_dataset_view(
            project_id=project.id,
            name="My view",
            media_ids=[UUID(media["image1"].id), UUID(media["image2"].id), UUID(media["video1"].id)],
        )

        statistics = fxt_dataset_view_service.get_dataset_view_statistics(
            project_id=project.id, dataset_view_id=view.id
        )

        assert statistics.media_counts.images == 2
        assert statistics.media_counts.videos == 1
        assert statistics.media_counts.video_frames == 100  # video1.frame_count
        assert statistics.annotations_counts.annotated_images == 1
        assert statistics.annotations_counts.annotated_videos == 1
        assert statistics.annotations_counts.instances == 2

    def test_get_dataset_view_statistics_empty_view(
        self, fxt_dataset_view_service: DatasetViewService, fxt_project_with_pipeline
    ):
        project, _ = fxt_project_with_pipeline
        view = fxt_dataset_view_service.create_dataset_view(project_id=project.id, name="Empty view")

        statistics = fxt_dataset_view_service.get_dataset_view_statistics(
            project_id=project.id, dataset_view_id=view.id
        )

        assert statistics.media_counts.images == 0
        assert statistics.media_counts.videos == 0
        assert statistics.annotations_counts.instances == 0

    def test_statistics_dataset_view_not_found(
        self, fxt_dataset_view_service: DatasetViewService, fxt_project_with_pipeline
    ):
        project, _ = fxt_project_with_pipeline

        with pytest.raises(ResourceNotFoundError):
            fxt_dataset_view_service.get_dataset_view_statistics(project_id=project.id, dataset_view_id=uuid4())
