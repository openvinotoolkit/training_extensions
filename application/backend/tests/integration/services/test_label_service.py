# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from sqlalchemy.orm import Session

from app.db.schema import DatasetItemDB, DatasetItemLabelDB, LabelDB, MediaDB, ProjectDB
from app.models import DatasetItemAnnotation, Label
from app.models.label import LabelReference, LabelUpdateInfo
from app.models.media import ImageFormat, MediaType
from app.models.project import Project
from app.models.shape import FullImage, Rectangle
from app.models.task import Task, TaskType
from app.services import ResourceNotFoundError, ResourceType, ResourceWithIdAlreadyExistsError
from app.services.dataset_service import DatasetService
from app.services.event.event_bus import EventType
from app.services.label_service import DuplicateLabelsError, LabelService


@pytest.fixture
def fxt_stored_project_with_labels(
    fxt_db_projects: list[ProjectDB],
    fxt_db_labels: list[LabelDB],
    db_session: Session,
) -> tuple[ProjectDB, list[LabelDB]]:
    db_project = fxt_db_projects[0]
    db_session.add(db_project)
    db_session.flush()

    for label in fxt_db_labels:
        label.project_id = db_project.id
    db_session.add_all(fxt_db_labels)
    db_session.flush()

    return db_project, fxt_db_labels


def _db_project_to_project(db_project: ProjectDB) -> Project:
    return Project(
        id=UUID(db_project.id),
        name=db_project.name,
        created_at=db_project.created_at,
        task=Task(
            task_type=TaskType(db_project.task_type),
            labels=[],
        ),
    )


@pytest.fixture
def fxt_db_images() -> list[MediaDB]:
    return [
        MediaDB(
            id=str(uuid4()),
            type=MediaType.IMAGE,
            project_id=str(uuid4()),
            name=f"test_image_{i}",
            format=ImageFormat.JPG,
            width=1024,
            height=768,
            size=2048,
        )
        for i in range(3)
    ]


class TestLabelServiceIntegration:
    def test_create_label(
        self,
        fxt_stored_project_with_labels: tuple[ProjectDB, list[LabelDB]],
        fxt_label_service: LabelService,
        db_session: Session,
    ) -> None:
        """
        Test creating a label for the project.

        Verifies that:
        - A new label can be created for the existing project
        """
        db_project, _ = fxt_stored_project_with_labels
        created_label = fxt_label_service.create_label(
            project_id=UUID(db_project.id), name="bird", color="#4500FF", hotkey="b"
        )

        assert created_label.name == "bird" and created_label.color == "#4500FF" and created_label.hotkey == "b"

        created_db_label = db_session.query(LabelDB).filter(LabelDB.id == str(created_label.id)).one()
        assert (
            created_db_label.name == "bird" and created_db_label.color == "#4500FF" and created_db_label.hotkey == "b"
        )

    def test_create_label_duplicate_name(
        self,
        fxt_stored_project_with_labels: tuple[ProjectDB, list[LabelDB]],
        fxt_label_service: LabelService,
    ) -> None:
        """
        Test creating a label with duplicating name for the project.

        Verifies that:
        - A new label with duplicating name cannot be created
        """
        db_project, _ = fxt_stored_project_with_labels

        with pytest.raises(DuplicateLabelsError):
            fxt_label_service.create_label(project_id=UUID(db_project.id), name="cat", color="#4500FF", hotkey="b")

    def test_create_label_duplicate_hotkey(
        self,
        fxt_stored_project_with_labels: tuple[ProjectDB, list[LabelDB]],
        fxt_label_service: LabelService,
    ) -> None:
        """
        Test creating a label with duplicating hotkey for the project.

        Verifies that:
        - A new label with duplicating hotkey cannot be created
        """
        db_project, _ = fxt_stored_project_with_labels

        with pytest.raises(DuplicateLabelsError):
            fxt_label_service.create_label(project_id=UUID(db_project.id), name="bird", color="#4500FF", hotkey="c")

    def test_create_label_duplicate_id(
        self,
        fxt_stored_project_with_labels: tuple[ProjectDB, list[LabelDB]],
        fxt_label_service: LabelService,
    ) -> None:
        """
        Test creating a label with duplicating ID for the project.

        Verifies that:
        - A new label with duplicating ID cannot be created
        """
        db_project, db_labels = fxt_stored_project_with_labels
        existing_label_id = db_labels[0].id

        with pytest.raises(ResourceWithIdAlreadyExistsError) as exc_info:
            fxt_label_service.create_label(
                project_id=UUID(db_project.id),
                label_id=UUID(existing_label_id),
                name="bird",
                color="#4500FF",
                hotkey="b",
            )

        assert exc_info.value.resource_type == ResourceType.LABEL
        assert exc_info.value.resource_id == existing_label_id

    def test_update_labels_add_edit_delete(
        self,
        fxt_stored_project_with_labels: tuple[ProjectDB, list[LabelDB]],
        fxt_label_service: LabelService,
        db_session: Session,
    ) -> None:
        """
        Test combined add, edit, and delete via update_labels.

        Verifies that:
        - All three operations are applied in a single call
        """
        db_project, db_labels = fxt_stored_project_with_labels
        project = _db_project_to_project(db_project)
        edit_id = UUID(db_labels[0].id)
        remove_id = UUID(db_labels[1].id)
        new_label_id = uuid4()

        result = fxt_label_service.update_labels(
            project=project,
            labels_to_add=[Label(id=new_label_id, project_id=project.id, name="bird", color="#0000FF", hotkey="b")],
            labels_to_remove=[LabelReference(id=remove_id)],
            labels_to_edit=[LabelUpdateInfo(id=edit_id, new_name="updated_cat", new_color="#121212", new_hotkey=None)],
        )

        result_ids = {lbl.id for lbl in result}
        assert edit_id in result_ids
        assert remove_id not in result_ids
        assert new_label_id in result_ids

        edited = next(lbl for lbl in result if lbl.id == edit_id)
        assert edited.name == "updated_cat" and edited.color == "#121212"

        added = next(lbl for lbl in result if lbl.id == new_label_id)
        assert added.name == "bird" and added.color == "#0000FF" and added.hotkey == "b"

        assert db_session.query(LabelDB).filter(LabelDB.id == str(remove_id)).one_or_none() is None

    def test_update_labels_recolor_notifies_inference(
        self,
        fxt_stored_project_with_labels: tuple[ProjectDB, list[LabelDB]],
        fxt_label_service: LabelService,
        fxt_event_bus: MagicMock,
        db_session: Session,
    ) -> None:
        """
        Test that changing a label colour notifies the inference pipeline after the commit.

        Without this notification, the running pipeline keeps rendering predictions with the
        colours cached when the model was last loaded.
        """
        db_project, db_labels = fxt_stored_project_with_labels
        project = _db_project_to_project(db_project)
        edit_id = UUID(db_labels[0].id)

        fxt_label_service.update_labels(
            project=project,
            labels_to_add=[],
            labels_to_remove=[],
            labels_to_edit=[
                LabelUpdateInfo(id=edit_id, new_name=db_labels[0].name, new_color="#121212", new_hotkey=None)
            ],
        )

        fxt_event_bus.emit_event_after_commit.assert_called_once_with(db_session, EventType.INFERENCE_PARAMS_CHANGED)

    def test_update_labels_without_color_change_does_not_notify_inference(
        self,
        fxt_stored_project_with_labels: tuple[ProjectDB, list[LabelDB]],
        fxt_label_service: LabelService,
        fxt_event_bus: MagicMock,
    ) -> None:
        """
        Test that edits which leave the label colours untouched do not notify the inference pipeline.
        """
        db_project, db_labels = fxt_stored_project_with_labels
        project = _db_project_to_project(db_project)
        edit_id = UUID(db_labels[0].id)
        unchanged_color = db_labels[0].color

        fxt_label_service.update_labels(
            project=project,
            labels_to_add=[],
            labels_to_remove=[],
            labels_to_edit=[
                LabelUpdateInfo(id=edit_id, new_name="renamed_cat", new_color=unchanged_color, new_hotkey=None)
            ],
        )

        fxt_event_bus.emit_event_after_commit.assert_not_called()

    def test_update_labels_edit_duplicate_name(
        self,
        fxt_stored_project_with_labels: tuple[ProjectDB, list[LabelDB]],
        fxt_label_service: LabelService,
    ) -> None:
        """
        Test editing a label to have a duplicate name raises DuplicateLabelsError.
        """
        db_project, db_labels = fxt_stored_project_with_labels
        project = _db_project_to_project(db_project)

        with pytest.raises(DuplicateLabelsError):
            fxt_label_service.update_labels(
                project=project,
                labels_to_add=[],
                labels_to_remove=[],
                labels_to_edit=[
                    LabelUpdateInfo(
                        id=UUID(db_labels[0].id),
                        new_name=db_labels[1].name,
                        new_color="#4500FF",
                        new_hotkey="b",
                    )
                ],
            )

    def test_update_labels_edit_duplicate_hotkey(
        self,
        fxt_stored_project_with_labels: tuple[ProjectDB, list[LabelDB]],
        fxt_label_service: LabelService,
    ) -> None:
        """
        Test editing a label to have a duplicate hotkey raises DuplicateLabelsError.
        """
        db_project, db_labels = fxt_stored_project_with_labels
        project = _db_project_to_project(db_project)

        with pytest.raises(DuplicateLabelsError):
            fxt_label_service.update_labels(
                project=project,
                labels_to_add=[],
                labels_to_remove=[],
                labels_to_edit=[
                    LabelUpdateInfo(
                        id=UUID(db_labels[0].id),
                        new_name="bird",
                        new_color="#4500FF",
                        new_hotkey=db_labels[1].hotkey,
                    )
                ],
            )

    def test_update_labels_edit_nonexistent(
        self,
        fxt_stored_project_with_labels: tuple[ProjectDB, list[LabelDB]],
        fxt_label_service: LabelService,
    ) -> None:
        """
        Test editing a non-existent label raises ResourceNotFoundError.
        """
        db_project, _ = fxt_stored_project_with_labels
        project = _db_project_to_project(db_project)
        nonexistent_id = uuid4()

        with pytest.raises(ResourceNotFoundError) as exc_info:
            fxt_label_service.update_labels(
                project=project,
                labels_to_add=[],
                labels_to_remove=[],
                labels_to_edit=[
                    LabelUpdateInfo(id=nonexistent_id, new_name="bird", new_color="#4500FF", new_hotkey="b")
                ],
            )

        assert exc_info.value.resource_type == ResourceType.LABEL
        assert exc_info.value.resource_id == str(nonexistent_id)

    def test_update_labels_delete_nonexistent(
        self,
        fxt_stored_project_with_labels: tuple[ProjectDB, list[LabelDB]],
        fxt_label_service: LabelService,
    ) -> None:
        """
        Test deleting a non-existent label raises ResourceNotFoundError.
        """
        db_project, _ = fxt_stored_project_with_labels
        project = _db_project_to_project(db_project)
        nonexistent_id = uuid4()

        with pytest.raises(ResourceNotFoundError) as exc_info:
            fxt_label_service.update_labels(
                project=project,
                labels_to_add=[],
                labels_to_remove=[LabelReference(id=nonexistent_id)],
                labels_to_edit=[],
            )

        assert exc_info.value.resource_type == ResourceType.LABEL
        assert exc_info.value.resource_id == str(nonexistent_id)

    def test_update_labels_remove_all_raises_value_error(
        self,
        fxt_stored_project_with_labels: tuple[ProjectDB, list[LabelDB]],
        fxt_label_service: LabelService,
    ) -> None:
        """
        Test removing all labels raises ValueError (project requires at least one label).
        """
        db_project, db_labels = fxt_stored_project_with_labels
        project = _db_project_to_project(db_project)

        with pytest.raises(ValueError, match="A project requires at least one label"):
            fxt_label_service.update_labels(
                project=project,
                labels_to_add=[],
                labels_to_remove=[LabelReference(id=UUID(lbl.id)) for lbl in db_labels],
                labels_to_edit=[],
            )

    def test_update_labels_empty(
        self,
        fxt_stored_project_with_labels: tuple[ProjectDB, list[LabelDB]],
        fxt_label_service: LabelService,
    ) -> None:
        """
        Test update_labels with no changes returns all existing labels unchanged.
        """
        db_project, db_labels = fxt_stored_project_with_labels
        project = _db_project_to_project(db_project)

        result = fxt_label_service.update_labels(
            project=project,
            labels_to_add=[],
            labels_to_remove=[],
            labels_to_edit=[],
        )

        assert len(result) == len(db_labels)
        assert {lbl.id for lbl in result} == {UUID(lbl.id) for lbl in db_labels}

    def test_list_all(
        self,
        fxt_stored_project_with_labels: tuple[ProjectDB, list[LabelDB]],
        fxt_label_service: LabelService,
    ) -> None:
        """
        Test getting a list of labels for the project.

        Verifies that:
        - Stored project labels can be obtained successfully
        """
        db_project, db_labels = fxt_stored_project_with_labels

        labels = fxt_label_service.list_all(UUID(db_project.id))

        assert len(labels) == len(db_labels)
        assert [label.name for label in labels] == [db_label.name for db_label in db_labels]

    def test_list_all_non_existing_project(
        self,
        fxt_stored_project_with_labels: tuple[ProjectDB, list[LabelDB]],
        fxt_label_service: LabelService,
    ) -> None:
        """
        Test getting a list of labels for non-existing project.

        Verifies that:
        - In case of non-existing project empty labels list is returned
        """
        fxt_stored_project_with_labels  # ensure fixture runs

        labels = fxt_label_service.list_all(uuid4())

        assert len(labels) == 0

    def test_list_ids(
        self,
        fxt_stored_project_with_labels: tuple[ProjectDB, list[LabelDB]],
        fxt_label_service: LabelService,
    ) -> None:
        """
        Test getting a list of labels ID's for the project.

        Verifies that:
        - Stored project labels ID's can be obtained successfully
        """
        db_project, db_labels = fxt_stored_project_with_labels

        ids = fxt_label_service.list_ids(UUID(db_project.id))

        assert len(ids) == len(db_labels)
        assert ids == [UUID(db_label.id) for db_label in db_labels]

    def test_list_ids_non_existing_project(
        self,
        fxt_stored_project_with_labels: tuple[ProjectDB, list[LabelDB]],
        fxt_label_service: LabelService,
    ) -> None:
        """
        Test getting a list of labels ID's for non-existing project.

        Verifies that:
        - In case of non-existing project empty labels ID's list is returned
        """
        fxt_stored_project_with_labels  # ensure fixture runs

        ids = fxt_label_service.list_ids(uuid4())

        assert len(ids) == 0

    def test_delete_label_removes_annotations_detection(
        self,
        fxt_stored_project_with_labels: tuple[ProjectDB, list[LabelDB]],
        fxt_label_service: LabelService,
        fxt_dataset_service: DatasetService,
        fxt_db_images: list[MediaDB],
        db_session: Session,
    ) -> None:
        """
        Test deleting a label cleans up annotations for a detection project.

        Sets up three dataset items:
        - item1: has two shapes, one with the deleted label only (dropped),
          the other with a different label (kept) -> annotation_data has one shape left.
        - item2: has one shape with only the deleted label -> annotation_data becomes [].
        - item3: has one shape with only the other label -> annotation_data is unchanged.

        Verifies that:
        - Shapes that lose all labels are removed.
        - Items that lose all shapes get annotation_data=[].
        - Items with remaining shapes keep them intact.
        - The label is removed from the labels join table.
        - The label itself is deleted.
        """
        db_project, db_labels = fxt_stored_project_with_labels
        project = _db_project_to_project(db_project)
        label_to_delete = db_labels[0]
        label_to_keep = db_labels[1]

        # Store media directly in the DB (no file I/O needed)
        for media in fxt_db_images:
            media.project_id = db_project.id
            db_session.add(media)
        db_session.flush()

        def _media(db_media: MediaDB) -> MagicMock:
            m = MagicMock()
            m.id = UUID(db_media.id)
            m.width = db_media.width
            m.height = db_media.height
            return m

        # item1: two rectangles — one with the deleted label, one with the kept label
        item1 = fxt_dataset_service.create_dataset_item(
            project_id=UUID(db_project.id),
            task=project.task,
            media=_media(fxt_db_images[0]),
            user_reviewed=True,
            annotations=[
                DatasetItemAnnotation(
                    shape=Rectangle(x=0, y=0, width=10, height=10),
                    labels=[LabelReference(id=UUID(label_to_delete.id))],
                ),
                DatasetItemAnnotation(
                    shape=Rectangle(x=20, y=20, width=10, height=10),
                    labels=[LabelReference(id=UUID(label_to_keep.id))],
                ),
            ],
        )
        # item2: one rectangle with only the deleted label
        item2 = fxt_dataset_service.create_dataset_item(
            project_id=UUID(db_project.id),
            task=project.task,
            media=_media(fxt_db_images[1]),
            user_reviewed=True,
            annotations=[
                DatasetItemAnnotation(
                    shape=Rectangle(x=0, y=0, width=10, height=10),
                    labels=[LabelReference(id=UUID(label_to_delete.id))],
                ),
            ],
        )
        # item3: one rectangle with only the kept label (unaffected)
        item3 = fxt_dataset_service.create_dataset_item(
            project_id=UUID(db_project.id),
            task=project.task,
            media=_media(fxt_db_images[2]),
            user_reviewed=True,
            annotations=[
                DatasetItemAnnotation(
                    shape=Rectangle(x=0, y=0, width=10, height=10),
                    labels=[LabelReference(id=UUID(label_to_keep.id))],
                ),
            ],
        )

        # Add a third label so we can remove one and still have >= 2 (multiclass min) / >= 1
        fxt_label_service.create_label(project_id=UUID(db_project.id), name="extra", color="#999999", hotkey="e")

        # Delete the label via update_labels
        fxt_label_service.update_labels(
            project=project,
            labels_to_add=[],
            labels_to_remove=[LabelReference(id=UUID(label_to_delete.id))],
            labels_to_edit=[],
        )

        db_session.expire_all()

        # Verify the label is deleted
        assert db_session.query(LabelDB).filter(LabelDB.id == label_to_delete.id).one_or_none() is None

        # item1: should have one shape left (the one with label_to_keep)
        refreshed_item1 = db_session.query(DatasetItemDB).filter(DatasetItemDB.id == str(item1.id)).one()
        assert refreshed_item1.annotation_data is not None
        assert len(refreshed_item1.annotation_data) == 1
        assert refreshed_item1.annotation_data[0]["labels"][0]["id"] == label_to_keep.id
        assert refreshed_item1.user_reviewed is True

        # item2: should have empty annotation_data (not None, since this is detection)
        refreshed_item2 = db_session.query(DatasetItemDB).filter(DatasetItemDB.id == str(item2.id)).one()
        assert refreshed_item2.annotation_data == []
        assert refreshed_item2.user_reviewed is True

        # item3: should be unchanged
        refreshed_item3 = db_session.query(DatasetItemDB).filter(DatasetItemDB.id == str(item3.id)).one()
        assert refreshed_item3.annotation_data is not None
        assert len(refreshed_item3.annotation_data) == 1
        assert refreshed_item3.annotation_data[0]["labels"][0]["id"] == label_to_keep.id

        # Verify label is removed from dataset_items_labels join table
        remaining_join_entries = (
            db_session.query(DatasetItemLabelDB).filter(DatasetItemLabelDB.label_id == label_to_delete.id).all()
        )
        assert len(remaining_join_entries) == 0

    def test_delete_label_removes_annotations_multiclass_classification(
        self,
        fxt_db_projects: list[ProjectDB],
        fxt_label_service: LabelService,
        fxt_dataset_service: DatasetService,
        fxt_db_images: list[MediaDB],
        db_session: Session,
    ) -> None:
        """
        Test deleting a label cleans up annotations for a multi-class classification project.

        Sets up two dataset items:
        - item1: has one shape with only the deleted label -> annotation_data becomes None,
          user_reviewed becomes False (multi-class classification behavior).
        - item2: has one shape with both the deleted label and a kept label ->
          annotation_data keeps the shape with the remaining label.

        Verifies that:
        - Multi-class classification items that lose all shapes get annotation_data=None
          and user_reviewed=False.
        - Items with remaining labels keep their annotation data.
        - The label is removed from the labels join table.
        - The label itself is deleted.
        """
        # Use a classification project
        db_project = fxt_db_projects[1]  # classification project
        db_project.task_type = TaskType.CLASSIFICATION.value
        db_session.add(db_project)
        db_session.flush()

        # Create labels for this project
        label1 = LabelDB(project_id=db_project.id, name="cat", color="#FF0000", hotkey="c")
        label2 = LabelDB(project_id=db_project.id, name="dog", color="#00FF00", hotkey="d")
        label3 = LabelDB(project_id=db_project.id, name="bird", color="#0000FF", hotkey="b")
        db_session.add_all([label1, label2, label3])
        db_session.flush()

        # Store media directly in the DB (no file I/O needed)
        for media in fxt_db_images:
            media.project_id = db_project.id
            db_session.add(media)
        db_session.flush()

        # Build the project model with exclusive_labels=True for multi-class
        project = Project(
            id=UUID(db_project.id),
            name=db_project.name,
            created_at=db_project.created_at,
            task=Task(task_type=TaskType.CLASSIFICATION, labels=[], exclusive_labels=True),
        )

        def _media(db_media: MediaDB) -> MagicMock:
            m = MagicMock()
            m.id = UUID(db_media.id)
            m.width = db_media.width
            m.height = db_media.height
            return m

        # item1: one full_image annotation with only the label to delete.
        # DatasetService validates multiclass correctly (one label per annotation).
        item1 = fxt_dataset_service.create_dataset_item(
            project_id=UUID(db_project.id),
            task=project.task,
            media=_media(fxt_db_images[0]),
            user_reviewed=True,
            annotations=[
                DatasetItemAnnotation(
                    shape=FullImage(),
                    labels=[LabelReference(id=UUID(label1.id))],
                ),
            ],
        )

        # item2: one full_image annotation with both label1 and label2.
        # A multilabel task is used for creation because multiclass forbids >1 label per annotation.
        # The join table is populated explicitly afterwards to match the actual annotation_data.
        multilabel_task = Task(task_type=TaskType.CLASSIFICATION, labels=[], exclusive_labels=False)
        item2 = fxt_dataset_service.create_dataset_item(
            project_id=UUID(db_project.id),
            task=multilabel_task,
            media=_media(fxt_db_images[1]),
            user_reviewed=True,
            annotations=[
                DatasetItemAnnotation(
                    shape=FullImage(),
                    labels=[LabelReference(id=UUID(label1.id)), LabelReference(id=UUID(label2.id))],
                ),
            ],
        )

        # Delete label1 using the multiclass project
        fxt_label_service.update_labels(
            project=project,
            labels_to_add=[],
            labels_to_remove=[LabelReference(id=UUID(label1.id))],
            labels_to_edit=[],
        )

        db_session.expire_all()

        # Verify label is deleted
        assert db_session.query(LabelDB).filter(LabelDB.id == label1.id).one_or_none() is None

        # item1: multi-class with no shapes left -> annotation_data=None, user_reviewed=False
        refreshed_item1 = db_session.query(DatasetItemDB).filter(DatasetItemDB.id == str(item1.id)).one()
        assert refreshed_item1.annotation_data is None
        assert refreshed_item1.user_reviewed is False

        # item2: should still have the shape with label2
        refreshed_item2 = db_session.query(DatasetItemDB).filter(DatasetItemDB.id == str(item2.id)).one()
        assert refreshed_item2.annotation_data is not None
        assert len(refreshed_item2.annotation_data) == 1
        assert len(refreshed_item2.annotation_data[0]["labels"]) == 1
        assert refreshed_item2.annotation_data[0]["labels"][0]["id"] == label2.id
        assert refreshed_item2.user_reviewed is True

        # Verify label is removed from join table
        remaining_join_entries = (
            db_session.query(DatasetItemLabelDB).filter(DatasetItemLabelDB.label_id == label1.id).all()
        )
        assert len(remaining_join_entries) == 0
