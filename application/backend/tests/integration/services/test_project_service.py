# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from uuid import UUID, uuid4

import numpy as np
import pytest
from PIL import Image
from sqlalchemy.orm import Session

from app.db.schema import LabelDB, MediaDB, ModelRevisionDB, PipelineDB, ProjectDB
from app.models import Label, Task, TaskType, TrainingStatus
from app.models.media import ImageFormat
from app.services import ResourceWithIdAlreadyExistsError
from app.services.base import ResourceInUseError, ResourceNotFoundError, ResourceType
from app.services.label_service import DuplicateLabelsError
from app.services.media_service import ImageMetadata, MediaService
from app.services.project_service import ProjectService


class TestProjectServiceIntegration:
    """Integration tests for ProjectService."""

    def test_create_project(self, fxt_project_service: ProjectService, db_session: Session):
        """Test creating a project."""
        project_id = uuid4()
        labels = [
            Label(id=uuid4(), project_id=project_id, name="cat", color="#00FF00", hotkey="c"),
            Label(id=uuid4(), project_id=project_id, name="dog", color="#FF0000", hotkey="d"),
        ]
        task = Task(
            task_type=TaskType.CLASSIFICATION,
            exclusive_labels=True,
            labels=labels,
        )
        created_project = fxt_project_service.create_project(project_id, "Test Project", task)
        assert created_project.id is not None
        assert created_project.name == "Test Project"
        assert created_project.task.task_type == task.task_type
        assert len(created_project.task.labels) == 2
        assert {label.id for label in created_project.task.labels} == {label.id for label in labels}
        assert {label.name for label in created_project.task.labels} == {"cat", "dog"}
        assert {label.color for label in created_project.task.labels} == {"#00FF00", "#FF0000"}
        assert {label.hotkey for label in created_project.task.labels} == {"c", "d"}
        # Ensure the associated pipeline is created
        db_pipeline = db_session.get(PipelineDB, str(created_project.id))
        assert db_pipeline is not None
        assert not db_pipeline.is_running
        # Ensure labels are created
        db_labels = db_session.query(LabelDB).filter(LabelDB.project_id == str(created_project.id)).all()
        assert len(db_labels) == 2
        assert {label.name for label in db_labels} == {"cat", "dog"}

    def test_create_project_duplicate_labels(self, fxt_project_service: ProjectService, db_session: Session):
        """Test creating a project with duplicated labels."""
        project_id = uuid4()
        labels = [
            Label(id=uuid4(), project_id=project_id, name="cat", color="#00FF00", hotkey="c"),
            Label(id=uuid4(), project_id=project_id, name="cat", color="#FF0000", hotkey="d"),
        ]
        task = Task(
            task_type=TaskType.CLASSIFICATION,
            exclusive_labels=True,
            labels=labels,
        )
        with pytest.raises(DuplicateLabelsError):
            fxt_project_service.create_project(project_id, "Test Project", task)
        # Ensure the transaction is rolled back
        assert not db_session.is_active

    def test_create_project_duplicate_id(self, fxt_project_service: ProjectService, db_session: Session):
        """Test creating a project with duplicated ID."""
        db_project = ProjectDB(name="existing_project", task_type=TaskType.CLASSIFICATION, exclusive_labels=False)
        db_session.add(db_project)
        db_session.flush()

        project_id = db_project.id
        labels = [
            Label(id=uuid4(), project_id=UUID(project_id), name="cat", color="#00FF00", hotkey="c"),
            Label(id=uuid4(), project_id=UUID(project_id), name="dog", color="#FF0000", hotkey="d"),
        ]
        task = Task(
            task_type=TaskType.CLASSIFICATION,
            exclusive_labels=True,
            labels=labels,
        )
        with pytest.raises(ResourceWithIdAlreadyExistsError) as exc_info:
            fxt_project_service.create_project(UUID(project_id), "Test Project", task)

        assert exc_info.value.resource_type == ResourceType.PROJECT
        assert exc_info.value.resource_id == project_id

    def test_list_projects(
        self, fxt_project_service: ProjectService, fxt_db_projects: list[ProjectDB], db_session: Session
    ):
        """Test listing projects."""
        # Create projects first
        label_config = [
            {"name": "cat", "color": "#00FF00", "hotkey": "c"},
            {"name": "dog", "color": "#FF0000", "hotkey": "d"},
        ]
        for db_project in fxt_db_projects:
            db_session.add(db_project)
            db_session.flush()

            for config in label_config:
                label = LabelDB(**config)
                label.project_id = db_project.id
                db_session.add(label)
            db_session.flush()

        projects = fxt_project_service.list_projects()
        assert len(projects) == 3
        for i in range(3):
            assert projects[i].name == fxt_db_projects[i].name
            assert projects[i].task.task_type == fxt_db_projects[i].task_type
            assert len(projects[i].task.labels) == len(label_config)
            for j in range(len(label_config)):
                project_label = projects[i].task.labels[j]
                config = label_config[j]
                assert (
                    project_label.name == config["name"]
                    and project_label.color == config["color"]
                    and project_label.hotkey == config["hotkey"]
                )

    def test_get_project_by_id(
        self, fxt_project_service: ProjectService, fxt_db_projects: list[ProjectDB], db_session: Session
    ):
        """Test retrieving a project by ID."""
        db_project = fxt_db_projects[0]
        db_session.add(db_project)
        db_session.flush()

        fetched_project = fxt_project_service.get_project_by_id(UUID(db_project.id))
        assert str(fetched_project.id) == db_project.id
        assert fetched_project.name == db_project.name

    def test_get_trained_model_counts(self, fxt_project_service: ProjectService, db_session: Session):
        """Test counting successfully trained models per project."""
        projects = [
            ProjectDB(id=str(uuid4()), name=f"P{i}", task_type=TaskType.DETECTION, exclusive_labels=True)
            for i in range(3)
        ]
        db_session.add_all(projects)
        db_session.flush()
        statuses_per_project = [
            [TrainingStatus.SUCCESSFUL, TrainingStatus.SUCCESSFUL, TrainingStatus.FAILED],
            [TrainingStatus.IN_PROGRESS, TrainingStatus.NOT_STARTED],
            [],
        ]
        for project, statuses in zip(projects, statuses_per_project):
            db_session.add_all(
                ModelRevisionDB(
                    id=str(uuid4()),
                    project_id=project.id,
                    name="model",
                    architecture="object-detection-yolox-s",
                    training_status=training_status,
                    training_configuration={},
                    label_schema_revision={},
                )
                for training_status in statuses
            )
        db_session.flush()

        assert fxt_project_service.get_trained_model_counts() == {UUID(projects[0].id): 2}
        assert fxt_project_service.get_trained_model_counts([UUID(projects[1].id)]) == {}

    def test_get_project_thumbnail(self, fxt_project_service: ProjectService, db_session: Session):
        """Test retrieving a project returns correct thumbnail ID."""
        # First create a project
        db_project = ProjectDB(
            id=str(uuid4()),
            name="P1",
            task_type=TaskType.CLASSIFICATION,
            exclusive_labels=True,
        )
        db_session.add(db_project)
        db_session.flush()

        # No dataset items yet, should return None
        fetched_thumbnail_path = fxt_project_service.get_project_thumbnail_path(UUID(db_project.id))
        assert fetched_thumbnail_path is None

        # Add a media
        db_media = MediaDB(
            id=str(uuid4()),
            type="image",
            project_id=db_project.id,
            name="item1",
            format="jpg",
            width=1920,
            height=1080,
            size=1024,
            source_id=None,
        )
        db_session.add(db_media)
        db_session.flush()

        # Now it should return the path to the thumbnail
        fetched_thumbnail_path = fxt_project_service.get_project_thumbnail_path(UUID(db_project.id))
        assert (
            fetched_thumbnail_path
            == fxt_project_service._projects_dir / f"{db_project.id}/dataset/{db_media.id}-thumb.jpg"
        )

    def test_get_project_by_id_not_found(self, fxt_project_service: ProjectService):
        """Test retrieving a non-existent project raises error."""
        non_existent_id = uuid4()
        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_project_service.get_project_by_id(non_existent_id)

        assert excinfo.value.resource_type == ResourceType.PROJECT
        assert excinfo.value.resource_id == str(non_existent_id)

    def test_delete_project(
        self,
        fxt_project_service: ProjectService,
        fxt_media_service: MediaService,
        fxt_db_projects: list[ProjectDB],
        db_session: Session,
    ):
        """Test deleting a project."""
        db_project = fxt_db_projects[0]
        db_session.add(db_project)
        db_session.flush()
        project = fxt_project_service.get_project_by_id(UUID(db_project.id))

        # Create a dummy image
        dummy_image = Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))
        # Create media using the service
        media = fxt_media_service.create_image(
            ImageMetadata(
                project_id=project.id,
                name="test_image.jpg",
                image_format=ImageFormat.JPG,
                data=dummy_image,
                source_id=None,
            )
        )
        project_folder = fxt_project_service._projects_dir / db_project.id
        media_file = project_folder / f"dataset/{media.id}.jpg"
        assert project_folder.exists()
        assert media_file.exists()

        fxt_project_service.delete_project_by_id(UUID(db_project.id))
        db_session.expire_all()  # Clear the session cache

        assert db_session.get(ProjectDB, db_project.id) is None
        # Ensure the associated pipeline is deleted
        assert db_session.get(PipelineDB, db_project.id) is None
        # Ensure the project directory has been removed
        assert not project_folder.exists()

    def test_delete_active_project(
        self,
        fxt_project_service: ProjectService,
        fxt_db_projects: list[ProjectDB],
        db_session: Session,
    ):
        """Test deleting a project which has a running pipeline."""
        db_project = fxt_db_projects[0]
        db_pipeline = PipelineDB(project_id=db_project.id)
        db_pipeline.is_running = True
        db_session.add(db_project)
        db_session.flush()
        db_session.add(db_pipeline)
        db_session.flush()

        with pytest.raises(ResourceInUseError) as excinfo:
            fxt_project_service.delete_project_by_id(UUID(db_project.id))

        assert excinfo.value.resource_type == ResourceType.PROJECT
        assert excinfo.value.resource_id == db_project.id
        assert "Cannot delete a project with a running pipeline." in str(excinfo.value)

    def test_delete_project_not_found(
        self, fxt_project_service: ProjectService, fxt_db_projects: list[ProjectDB], db_session: Session
    ):
        """Test deleting a non-existing project."""
        non_existent_id = uuid4()

        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_project_service.delete_project_by_id(non_existent_id)

        assert excinfo.value.resource_type == ResourceType.PROJECT
        assert excinfo.value.resource_id == str(non_existent_id)

    def test_update_project_name(
        self, fxt_project_service: ProjectService, fxt_db_projects: list[ProjectDB], db_session: Session
    ):
        """Test updating a project's name."""
        db_project = fxt_db_projects[0]
        db_session.add(db_project)
        db_session.flush()

        new_name = "Updated Project Name"
        updated_project = fxt_project_service.update_project_name(UUID(db_project.id), new_name)

        assert updated_project.name == new_name

    def test_update_project_name_not_found(self, fxt_project_service: ProjectService):
        """Test updating a non-existent project name raises error."""
        non_existent_id = uuid4()
        new_name = "New Project Name"
        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_project_service.update_project_name(non_existent_id, new_name)

        assert excinfo.value.resource_type == ResourceType.PROJECT
        assert excinfo.value.resource_id == str(non_existent_id)
