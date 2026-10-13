# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Endpoints for managing projects"""

import secrets
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Body, Depends, Query, status
from fastapi.exceptions import HTTPException
from fastapi.openapi.models import Example
from pydantic import ValidationError as PydanticValidationError
from starlette.responses import FileResponse

from app.api.dependencies import get_data_collector, get_label_service, get_project, get_project_service
from app.api.schemas import LabelView, ModelCountsView, PatchLabels, ProjectCreate, ProjectUpdateName, ProjectView
from app.api.validators import ProjectID
from app.models import Label, LabelReference, LabelUpdateInfo, Task
from app.models.project import Project
from app.services import LabelService, ProjectService, ResourceInUseError, ResourceWithIdAlreadyExistsError
from app.services.data_collect import DataCollector
from app.services.label_service import DuplicateLabelsError

router = APIRouter(prefix="/api/projects", tags=["Projects"])


def _random_color() -> str:
    """Return a random #RRGGBB hex color for label assignment when none is provided."""
    return f"#{secrets.token_hex(3).upper()}"


IncludeModelCounts = Annotated[
    bool, Query(description="Whether to include the `model_counts` field (e.g. number of trained models).")
]


def _to_project_view(project: Project, trained_model_counts: dict[UUID, int] | None) -> ProjectView:
    view = ProjectView.model_validate(project, from_attributes=True)
    if trained_model_counts is None:
        return view
    return view.model_copy(
        update={"model_counts": ModelCountsView(trained_models=trained_model_counts.get(view.id, 0))}
    )


CREATE_PROJECT_BODY_DESCRIPTION = """
Configuration for creating a new project. Specify the project name, problem type (classification, detection, 
segmentation), and label definitions.
"""
CREATE_PROJECT_BODY_EXAMPLES = {
    "classification": Example(
        summary="Classification project with exclusive labels",
        description="Configuration for a classification project that have 2 labels: cat and dog",
        value={
            "name": "animals",
            "task": {
                "task_type": "classification",
                "exclusive_labels": True,
                "labels": [{"name": "cat"}, {"name": "dog"}],
            },
        },
    ),
    "detection": Example(
        summary="Detection project with exclusive labels",
        description="Configuration for a detection project that have 3 labels: Chardonnay, Sauvignon Blanc and "
        "Cabernet Franc",
        value={
            "name": "grapes",
            "task": {
                "task_type": "detection",
                "exclusive_labels": True,
                "labels": [{"name": "Chardonnay"}, {"name": "Sauvignon Blanc"}, {"name": "Cabernet Franc"}],
            },
        },
    ),
    "instance_segmentation": Example(
        summary="Instance segmentation project with exclusive labels",
        description="Configuration for an instance segmentation project that have 2 labels: car and person",
        value={
            "name": "traffic",
            "task": {
                "task_type": "instance_segmentation",
                "exclusive_labels": True,
                "labels": [{"name": "car"}, {"name": "person"}],
            },
        },
    ),
}


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    response_model=ProjectView,
    responses={
        status.HTTP_201_CREATED: {"description": "Project successfully created"},
        status.HTTP_409_CONFLICT: {"description": "Project already exists or labels have duplicated names or hotkeys"},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"description": "Invalid request body"},
    },
)
def create_project(
    project_config: Annotated[
        ProjectCreate, Body(description=CREATE_PROJECT_BODY_DESCRIPTION, openapi_examples=CREATE_PROJECT_BODY_EXAMPLES)
    ],
    project_service: Annotated[ProjectService, Depends(get_project_service)],
) -> ProjectView:
    """Create and configure a new project"""
    try:
        task = Task(
            exclusive_labels=project_config.task.exclusive_labels,
            task_type=project_config.task.task_type,
            labels=[
                Label.model_validate({**label.model_dump(), "color": label.color or _random_color()})
                for label in project_config.task.labels
            ],
        )
        created_project = project_service.create_project(project_config.id, project_config.name, task)
        return ProjectView.model_validate(created_project, from_attributes=True)
    except PydanticValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=e.errors())
    except (ResourceWithIdAlreadyExistsError, DuplicateLabelsError) as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))


@router.get(
    "",
    response_model=list[ProjectView],
    responses={
        status.HTTP_200_OK: {"description": "List of available projects"},
    },
)
def list_projects(
    project_service: Annotated[ProjectService, Depends(get_project_service)],
    include_model_counts: IncludeModelCounts = False,
) -> list[ProjectView]:
    """List the available projects"""
    trained_model_counts = project_service.get_trained_model_counts() if include_model_counts else None
    return [_to_project_view(proj, trained_model_counts) for proj in project_service.list_projects()]


@router.get(
    "/{project_id}",
    response_model=ProjectView,
    responses={
        status.HTTP_200_OK: {"description": "Project found"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid project ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Project not found"},
    },
)
def get_project_by_id(
    project_id: ProjectID,
    project_service: Annotated[ProjectService, Depends(get_project_service)],
    include_model_counts: IncludeModelCounts = False,
) -> ProjectView:
    """Get info about a given project"""
    project = project_service.get_project_by_id(project_id)
    trained_model_counts = project_service.get_trained_model_counts([project_id]) if include_model_counts else None
    return _to_project_view(project, trained_model_counts)


@router.patch(
    "/{project_id}",
    response_model=ProjectView,
    responses={
        status.HTTP_200_OK: {"description": "Project name updated successfully"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid project ID or request body"},
        status.HTTP_404_NOT_FOUND: {"description": "Project not found"},
    },
)
def rename_project(
    project_id: ProjectID,
    project_update_name: Annotated[ProjectUpdateName, Body(description="Updated project name")],
    project_service: Annotated[ProjectService, Depends(get_project_service)],
) -> ProjectView:
    """Rename a project"""
    updated = project_service.update_project_name(project_id, project_update_name.name)
    return ProjectView.model_validate(updated, from_attributes=True)


@router.delete(
    "/{project_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        status.HTTP_204_NO_CONTENT: {
            "description": "Project successfully deleted",
        },
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid project ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Project not found"},
        status.HTTP_409_CONFLICT: {"description": "Project has a running pipeline and cannot be deleted"},
    },
)
def delete_project(
    project_id: ProjectID,
    project_service: Annotated[ProjectService, Depends(get_project_service)],
) -> None:
    """Delete a project. Project with a running pipeline cannot be deleted."""
    try:
        project_service.delete_project_by_id(project_id)
    except ResourceInUseError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))


@router.patch(
    "/{project_id}/labels",
    response_model=list[LabelView],
    responses={
        status.HTTP_200_OK: {"description": "Labels updated successfully"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid project ID or request body"},
        status.HTTP_404_NOT_FOUND: {"description": "Project or label not found"},
        status.HTTP_409_CONFLICT: {
            "description": "Label(s) already exists or have duplicated names or hotkeys, "
            "or minimum number of labels for project will be violated after update."
        },
    },
)
def update_labels(
    project: Annotated[Project, Depends(get_project)],
    labels: Annotated[
        PatchLabels,
        Body(
            description="Labels to add, remove or edit",
        ),
    ],
    label_service: Annotated[LabelService, Depends(get_label_service)],
) -> list[LabelView]:
    """Update labels for a given project"""
    try:
        updated_labels = label_service.update_labels(
            project=project,
            labels_to_add=[Label.model_validate(lbl, from_attributes=True) for lbl in labels.labels_to_add],
            labels_to_edit=[LabelUpdateInfo.model_validate(lbl, from_attributes=True) for lbl in labels.labels_to_edit],
            labels_to_remove=[
                LabelReference.model_validate(lbl, from_attributes=True) for lbl in labels.labels_to_remove
            ],
        )
    except (ResourceWithIdAlreadyExistsError, DuplicateLabelsError, ValueError) as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))

    return [LabelView.model_validate(label, from_attributes=True) for label in updated_labels]


@router.get(
    "/{project_id}/thumbnail",
    responses={
        status.HTTP_200_OK: {"description": "Project thumbnail found"},
        status.HTTP_204_NO_CONTENT: {"description": "No thumbnail available"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid project ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Project not found"},
    },
)
def get_project_thumbnail(
    project_id: ProjectID, project_service: Annotated[ProjectService, Depends(get_project_service)]
) -> FileResponse:
    """Get the project's thumbnail image"""
    thumbnail_path = project_service.get_project_thumbnail_path(project_id)
    if thumbnail_path:
        return FileResponse(path=thumbnail_path)
    raise HTTPException(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/{project_id}/pipeline:capture",
    responses={
        status.HTTP_200_OK: {"description": "Successfully marked next pipeline frame to be collected"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid project ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Project not found"},
    },
)
def capture_next_pipeline_frame(
    project_id: ProjectID,
    project_service: Annotated[ProjectService, Depends(get_project_service)],
    data_collector: Annotated[DataCollector, Depends(get_data_collector)],
) -> None:
    """Marks next pipeline frame to be collected"""
    project_service.get_project_by_id(project_id)
    data_collector.collect_next_frame()
