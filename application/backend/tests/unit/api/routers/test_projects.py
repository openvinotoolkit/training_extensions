# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os
import tempfile
from datetime import UTC, datetime
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi import status

from app.api.dependencies import get_data_collector, get_label_service
from app.api.schemas import LabelView, PatchLabels, ProjectView, TaskView
from app.api.schemas.label import LabelCreate, LabelEdit, LabelRemove
from app.models import Label, TaskType
from app.services import ResourceInUseError, ResourceNotFoundError, ResourceType, ResourceWithIdAlreadyExistsError
from app.services.data_collect import DataCollector
from app.services.label_service import DuplicateLabelsError, LabelService


@pytest.fixture
def fxt_project() -> ProjectView:
    return ProjectView(
        id=uuid4(),
        name="Test Project",
        active_pipeline=False,
        created_at=datetime.now(tz=UTC),
        task=TaskView(
            task_type=TaskType.CLASSIFICATION,
            exclusive_labels=True,
            labels=[
                LabelView(id=uuid4(), name="cat", color="#11AA22", hotkey="s"),
                LabelView(id=uuid4(), name="dog", color="#AA2233", hotkey="d"),
            ],
        ),
    )


@pytest.fixture
def fxt_data_collector(fxt_app) -> MagicMock:
    data_collector = MagicMock(spec=DataCollector)
    fxt_app.dependency_overrides[get_data_collector] = lambda: data_collector
    return data_collector


@pytest.fixture
def fxt_label_service(fxt_app):
    label_service = MagicMock(spec=LabelService)
    fxt_app.dependency_overrides[get_label_service] = lambda: label_service
    return label_service


class TestProjectEndpoints:
    @pytest.mark.parametrize(
        "http_method, service_method",
        [
            ("get", "get_project_by_id"),
            ("delete", "delete_project_by_id"),
        ],
    )
    def test_project_invalid_ids(self, http_method, service_method, fxt_project_service, fxt_client):
        response = getattr(fxt_client, http_method)("/api/projects/invalid-id")

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        getattr(fxt_project_service, service_method).assert_not_called()

    def test_list_project_success(self, fxt_project, fxt_project_service, fxt_client):
        fxt_project_service.list_projects.return_value = [fxt_project] * 2

        response = fxt_client.get("/api/projects")

        assert response.status_code == status.HTTP_200_OK
        assert len(response.json()) == 2
        fxt_project_service.list_projects.assert_called_once()

    def test_get_project_success(self, fxt_project, fxt_project_service, fxt_client):
        fxt_project_service.get_project_by_id.return_value = fxt_project

        response = fxt_client.get(f"/api/projects/{str(fxt_project.id)}")

        assert response.status_code == status.HTTP_200_OK
        fxt_project_service.get_project_by_id.assert_called_once_with(fxt_project.id)

    @pytest.mark.parametrize("exclude_attrs", [{}, {"id"}])
    def test_create_project_success(self, exclude_attrs, fxt_project, fxt_project_service, fxt_client):
        fxt_project_service.create_project.return_value = fxt_project

        response = fxt_client.post("/api/projects", json=fxt_project.model_dump(mode="json", exclude=exclude_attrs))

        assert response.status_code == status.HTTP_201_CREATED
        fxt_project_service.create_project.assert_called_once()

    @pytest.mark.parametrize("exclude_attrs", [{}, {"id"}])
    def test_create_project_exists(self, exclude_attrs, fxt_project_service, fxt_project, fxt_client):
        fxt_project_service.create_project.side_effect = ResourceWithIdAlreadyExistsError(
            resource_type=ResourceType.PROJECT, resource_id="new_id"
        )
        response = fxt_client.post("/api/projects", json=fxt_project.model_dump(mode="json", exclude=exclude_attrs))

        assert response.status_code == status.HTTP_409_CONFLICT
        fxt_project_service.create_project.assert_called_once()

    def test_create_project_invalid(self, fxt_project_service, fxt_client):
        response = fxt_client.post("/api/projects", json={"name": "New Pipeline", "attr": "invalid"})

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
        fxt_project_service.create_project.assert_not_called()


class TestCreateProjectInputValidationSecurity:
    """Regression tests for a null label color must not cause a 500 error."""

    @pytest.mark.parametrize(
        "label_payload",
        [
            {"name": "cat"},  # color omitted, defaults to None in LabelCreate
            {"name": "cat", "color": None},  # color explicitly null
        ],
    )
    def test_null_label_color_gets_random_default_assigned(
        self, label_payload, fxt_project_service, fxt_project, fxt_client
    ):
        """Label with color=null must return 201 and receive a valid hex color, not cause 500."""
        import re

        fxt_project_service.create_project.return_value = fxt_project
        payload = {
            "name": "Security Test Project",
            "task": {
                "task_type": "classification",
                "exclusive_labels": True,
                # classification + exclusive_labels requires >= 2 labels (TaskCreate.validate_labels)
                "labels": [
                    {"id": str(uuid4()), **label_payload},
                    {"id": str(uuid4()), "name": "dog", "color": "#001122"},
                ],
            },
        }

        response = fxt_client.post("/api/projects", json=payload)

        assert response.status_code == status.HTTP_201_CREATED
        fxt_project_service.create_project.assert_called_once()

        # Verify the handler assigned a valid #RRGGBB hex color before calling the service
        task_arg = fxt_project_service.create_project.call_args[0][2]
        hex_pattern = re.compile(r"^#[0-9A-Fa-f]{6}$")
        assert task_arg.labels, "expected at least one label"
        for label in task_arg.labels:
            assert label.color is not None, "color must not reach the service as None"
            assert hex_pattern.match(label.color), f"color must be #RRGGBB, got {label.color!r}"

    def test_provided_color_preserved(self, fxt_project_service, fxt_project, fxt_client):
        """An explicitly provided color must not be overridden."""
        fxt_project_service.create_project.return_value = fxt_project
        payload = {
            "name": "Color Project",
            "task": {
                "task_type": "classification",
                "exclusive_labels": True,
                # classification + exclusive_labels requires >= 2 labels
                "labels": [
                    {"id": str(uuid4()), "name": "cat", "color": "#AABB11"},
                    {"id": str(uuid4()), "name": "dog", "color": "#112233"},
                ],
            },
        }

        response = fxt_client.post("/api/projects", json=payload)

        assert response.status_code == status.HTTP_201_CREATED
        task_arg = fxt_project_service.create_project.call_args[0][2]
        assert task_arg.labels[0].color == "#AABB11"

    def test_delete_project_success(self, fxt_project, fxt_project_service, fxt_client):
        response = fxt_client.delete(f"/api/projects/{str(fxt_project.id)}")

        assert response.status_code == status.HTTP_204_NO_CONTENT
        fxt_project_service.delete_project_by_id.assert_called_once_with(fxt_project.id)

    def test_delete_project_not_found(self, fxt_project_service, fxt_client):
        project_id = str(uuid4())
        fxt_project_service.delete_project_by_id.side_effect = ResourceNotFoundError(ResourceType.PROJECT, project_id)

        response = fxt_client.delete(f"/api/projects/{project_id}")

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_delete_project_in_use(self, fxt_project, fxt_project_service, fxt_client):
        project_id = str(fxt_project.id)
        err = ResourceInUseError(ResourceType.PROJECT, project_id)
        fxt_project_service.delete_project_by_id.side_effect = err

        response = fxt_client.delete(f"/api/projects/{project_id}")

        assert response.status_code == status.HTTP_409_CONFLICT
        assert str(err) == response.json()["detail"]

    def test_update_labels_success(self, fxt_get_project, fxt_label_service, fxt_client):
        """Test successful label update with add, edit, and remove operations."""
        # Setup
        new_label_id = uuid4()
        labels_to_add = [LabelCreate(id=new_label_id, name="mouse", color="#0000FF", hotkey="m")]
        labels_to_edit = [
            LabelEdit(
                id=fxt_get_project.task.labels[0].id, new_name="updated_cat", new_color="#121212", new_hotkey=None
            )
        ]
        labels_to_remove = [LabelRemove(id=fxt_get_project.task.labels[1].id)]

        patch_labels = PatchLabels(
            labels_to_add=labels_to_add, labels_to_edit=labels_to_edit, labels_to_remove=labels_to_remove
        )

        fxt_label_service.update_labels.return_value = [
            Label(
                id=fxt_get_project.task.labels[0].id,
                project_id=fxt_get_project.id,
                name="updated_cat",
                color="#121212",
                hotkey=None,
            ),
            Label(id=new_label_id, project_id=fxt_get_project.id, name="mouse", color="#0000FF", hotkey="m"),
        ]

        # Execute
        response = fxt_client.patch(
            f"/api/projects/{str(fxt_get_project.id)}/labels", json=patch_labels.model_dump(mode="json")
        )

        # Assert
        assert response.status_code == status.HTTP_200_OK
        fxt_label_service.update_labels.assert_called_once()
        call_kwargs = fxt_label_service.update_labels.call_args[1]
        assert call_kwargs["project"].id == fxt_get_project.id

        assert len(call_kwargs["labels_to_add"]) == 1
        assert call_kwargs["labels_to_add"][0].id == new_label_id
        assert call_kwargs["labels_to_add"][0].name == "mouse"
        assert call_kwargs["labels_to_add"][0].color == "#0000FF"
        assert call_kwargs["labels_to_add"][0].hotkey == "m"

        assert len(call_kwargs["labels_to_edit"]) == 1
        assert call_kwargs["labels_to_edit"][0].id == fxt_get_project.task.labels[0].id
        assert call_kwargs["labels_to_edit"][0].new_name == "updated_cat"
        assert call_kwargs["labels_to_edit"][0].new_color == "#121212"
        assert call_kwargs["labels_to_edit"][0].new_hotkey is None

        assert len(call_kwargs["labels_to_remove"]) == 1
        assert call_kwargs["labels_to_remove"][0].id == fxt_get_project.task.labels[1].id

        assert response.json() == [
            {
                "id": str(fxt_get_project.task.labels[0].id),
                "name": "updated_cat",
                "color": "#121212",
                "hotkey": None,
            },
            {
                "id": str(new_label_id),
                "name": "mouse",
                "color": "#0000FF",
                "hotkey": "m",
            },
        ]

    def test_update_labels_empty(self, fxt_get_project, fxt_label_service, fxt_client):
        # Setup
        patch_labels = PatchLabels()

        fxt_label_service.update_labels.return_value = []

        # Execute
        response = fxt_client.patch(
            f"/api/projects/{str(fxt_get_project.id)}/labels", json=patch_labels.model_dump(mode="json")
        )

        # Assert
        assert response.status_code == status.HTTP_200_OK
        fxt_label_service.update_labels.assert_called_once()
        call_kwargs = fxt_label_service.update_labels.call_args[1]
        assert call_kwargs["labels_to_add"] == []
        assert call_kwargs["labels_to_edit"] == []
        assert call_kwargs["labels_to_remove"] == []

    def test_update_labels_duplicate_labels_error(self, fxt_get_project, fxt_label_service, fxt_client):
        """Test label update with duplicate attributes returns 409."""
        new_label_id = uuid4()
        labels_to_add = [LabelCreate(id=new_label_id, name="cat", color="#FF0000", hotkey="c")]
        patch_labels = PatchLabels(labels_to_add=labels_to_add)

        fxt_label_service.update_labels.side_effect = DuplicateLabelsError()

        response = fxt_client.patch(
            f"/api/projects/{str(fxt_get_project.id)}/labels", json=patch_labels.model_dump(mode="json")
        )

        assert response.status_code == status.HTTP_409_CONFLICT
        assert response.json()["detail"] == "Either label names or hotkeys have duplicates"

    @pytest.mark.parametrize(
        "patch_labels",
        [
            PatchLabels(labels_to_remove=[LabelRemove(id=uuid4())]),  # Non-existent label to remove
            PatchLabels(
                labels_to_edit=[LabelEdit(id=uuid4(), new_name="updated", new_color="#121212", new_hotkey=None)]
            ),  # Non-existent label to edit
        ],
    )
    def test_update_labels_remove_edit_nonexistent(self, patch_labels, fxt_get_project, fxt_label_service, fxt_client):
        """Test editing or removing labels that don't exist returns 404."""
        fxt_label_service.update_labels.side_effect = ResourceNotFoundError(
            resource_type=ResourceType.LABEL,
            resource_id=str(uuid4()),
            message="One or more labels to remove do not exist in the project",
        )

        response = fxt_client.patch(
            f"/api/projects/{str(fxt_get_project.id)}/labels", json=patch_labels.model_dump(mode="json")
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "do not exist" in response.json()["detail"]

    def test_update_labels_id_already_exists(self, fxt_get_project, fxt_label_service, fxt_client):
        """Test label update with existing label ID returns 409."""
        existing_label_id = uuid4()
        labels_to_add = [LabelCreate(id=existing_label_id, name="new_label", color="#FF0000", hotkey="n")]
        patch_labels = PatchLabels(labels_to_add=labels_to_add)

        fxt_label_service.update_labels.side_effect = ResourceWithIdAlreadyExistsError(
            resource_type=ResourceType.LABEL, resource_id=str(existing_label_id)
        )

        response = fxt_client.patch(
            f"/api/projects/{str(fxt_get_project.id)}/labels", json=patch_labels.model_dump(mode="json")
        )

        assert response.status_code == status.HTTP_409_CONFLICT

    def test_update_labels_value_error(self, fxt_get_project, fxt_label_service, fxt_client):
        """Test label update that violates label count constraints returns 409."""
        labels_to_remove = [LabelRemove(id=fxt_get_project.task.labels[0].id)]
        patch_labels = PatchLabels(labels_to_remove=labels_to_remove)

        fxt_label_service.update_labels.side_effect = ValueError(
            "Multi-class classifications requires at least two labels, but after this label update the total "
            "number of labels is 1."
        )

        response = fxt_client.patch(
            f"/api/projects/{str(fxt_get_project.id)}/labels", json=patch_labels.model_dump(mode="json")
        )

        assert response.status_code == status.HTTP_409_CONFLICT
        assert "at least two labels" in response.json()["detail"]

    def test_update_labels_invalid_project_id(self, fxt_project_service, fxt_client):
        """Test update with invalid project ID returns 400."""
        response = fxt_client.patch("/api/projects/invalid-id/labels", json={})

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        fxt_project_service.get_project_by_id.assert_not_called()

    def test_get_project_thumbnail(self, fxt_project, fxt_project_service, fxt_client):
        tmp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                tmp_file_path = tmp_file.name
                fxt_project_service.get_project_thumbnail_path.return_value = tmp_file_path
                response = fxt_client.get(f"/api/projects/{str(fxt_project.id)}/thumbnail")

            assert response.status_code == status.HTTP_200_OK
            fxt_project_service.get_project_thumbnail_path.assert_called_once()
        finally:
            if tmp_file_path and os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

    def test_get_project_thumbnail_none(self, fxt_project, fxt_project_service, fxt_client):
        fxt_project_service.get_project_thumbnail_path.return_value = None
        response = fxt_client.get(f"/api/projects/{str(fxt_project.id)}/thumbnail")
        assert response.status_code == status.HTTP_204_NO_CONTENT
        fxt_project_service.get_project_thumbnail_path.assert_called_once()

    def test_capture_next_pipeline_frame_invalid_project_id(self, fxt_project_service, fxt_data_collector, fxt_client):
        """Test capture next pipeline frame with invalid project ID returns 400."""
        response = fxt_client.post("/api/projects/invalid-id/pipeline:capture")

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        fxt_project_service.get_project_by_id.assert_not_called()

    def test_capture_next_pipeline_frame_project_not_found(self, fxt_project_service, fxt_data_collector, fxt_client):
        """Test capture next pipeline frame for non-existent project returns 404."""
        project_id = uuid4()
        fxt_project_service.get_project_by_id.side_effect = ResourceNotFoundError(ResourceType.PROJECT, str(project_id))

        response = fxt_client.post(f"/api/projects/{str(project_id)}/pipeline:capture")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_data_collector.collect_next_frame.assert_not_called()

    def test_capture_next_pipeline_frame(self, fxt_project, fxt_project_service, fxt_data_collector, fxt_client):
        """Test capture next pipeline frame for existing project returns 200."""
        project_id = uuid4()
        fxt_project_service.get_project_by_id.return_value = fxt_project

        response = fxt_client.post(f"/api/projects/{str(project_id)}/pipeline:capture")

        assert response.status_code == status.HTTP_200_OK
        fxt_data_collector.collect_next_frame.assert_called_once_with()
