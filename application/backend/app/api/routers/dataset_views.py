# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from typing import Annotated

from fastapi import APIRouter, Body, Depends, status

from app.api.dependencies import get_dataset_view, get_dataset_view_service, get_project
from app.api.schemas.dataset_view import (
    AssignMediaToDatasetView,
    DatasetViewCreate,
    DatasetViewUpdateName,
    DatasetViewView,
    UnassignMediaFromDatasetView,
)
from app.models import Project
from app.models.dataset_view import DatasetView
from app.services import DatasetViewService

router = APIRouter(prefix="/api/projects/{project_id}/dataset/views", tags=["Dataset Views"])


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_201_CREATED: {"description": "Dataset view successfully created", "model": DatasetViewView},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid project ID or request body"},
        status.HTTP_404_NOT_FOUND: {"description": "Project not found"},
        status.HTTP_409_CONFLICT: {"description": "A dataset view with the same name already exists"},
    },
)
def create_dataset_view(
    project: Annotated[Project, Depends(get_project)],
    dataset_view_create: Annotated[DatasetViewCreate, Body()],
    dataset_view_service: Annotated[DatasetViewService, Depends(get_dataset_view_service)],
) -> DatasetViewView:
    """
    Create a new, named dataset view. The view may be created empty or pre-populated with selected media.

    Note: the content of a dataset view is not listed through this section of the API. To list the media assigned
    to a view, use `GET /api/projects/{project_id}/dataset/media?dataset_view_id={dataset_view_id}`; the
    corresponding dataset items and statistics are available through `GET /api/projects/{project_id}/dataset/items`
    and `GET /api/projects/{project_id}/dataset/statistics` with the same `dataset_view_id` query parameter.
    """
    dataset_view = dataset_view_service.create_dataset_view(
        project_id=project.id,
        name=dataset_view_create.name,
        media_ids=dataset_view_create.media_ids,
    )
    return DatasetViewView.model_validate(dataset_view, from_attributes=True)


@router.get(
    "",
    response_model=list[DatasetViewView],
    responses={
        status.HTTP_200_OK: {"description": "List of available dataset views"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid project ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Project not found"},
    },
)
def list_dataset_views(
    project: Annotated[Project, Depends(get_project)],
    dataset_view_service: Annotated[DatasetViewService, Depends(get_dataset_view_service)],
) -> list[DatasetViewView]:
    """
    List the dataset views defined in a project.

    This endpoint returns the views themselves (id, name, ...), not their content. To list the media assigned to
    a specific view, use `GET /api/projects/{project_id}/dataset/media?dataset_view_id={dataset_view_id}`.
    """
    return [
        DatasetViewView.model_validate(dataset_view, from_attributes=True)
        for dataset_view in dataset_view_service.list_dataset_views(project_id=project.id)
    ]


@router.get(
    "/{dataset_view_id}",
    responses={
        status.HTTP_200_OK: {"description": "Dataset view found", "model": DatasetViewView},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid project or dataset view ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Project or dataset view not found"},
    },
)
def get_dataset_view_details(
    dataset_view: Annotated[DatasetView, Depends(get_dataset_view)],
) -> DatasetViewView:
    """
    Get information about a specific dataset view.

    The media assigned to the view are not part of the response; they can be listed with
    `GET /api/projects/{project_id}/dataset/media?dataset_view_id={dataset_view_id}`.
    """
    return DatasetViewView.model_validate(dataset_view, from_attributes=True)


@router.patch(
    "/{dataset_view_id}",
    responses={
        status.HTTP_200_OK: {"description": "Dataset view successfully renamed", "model": DatasetViewView},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid project or dataset view ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Project or dataset view not found"},
        status.HTTP_409_CONFLICT: {"description": "A dataset view with the same name already exists"},
    },
)
def rename_dataset_view(
    project: Annotated[Project, Depends(get_project)],
    dataset_view: Annotated[DatasetView, Depends(get_dataset_view)],
    dataset_view_update_name: Annotated[DatasetViewUpdateName, Body(description="Updated dataset view name")],
    dataset_view_service: Annotated[DatasetViewService, Depends(get_dataset_view_service)],
) -> DatasetViewView:
    """Rename a dataset view."""
    updated = dataset_view_service.rename_dataset_view(
        project_id=project.id, dataset_view_id=dataset_view.id, new_name=dataset_view_update_name.name
    )
    return DatasetViewView.model_validate(updated, from_attributes=True)


@router.delete(
    "/{dataset_view_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        status.HTTP_204_NO_CONTENT: {"description": "Dataset view successfully deleted"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid project or dataset view ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Project or dataset view not found"},
    },
)
def delete_dataset_view(
    project: Annotated[Project, Depends(get_project)],
    dataset_view: Annotated[DatasetView, Depends(get_dataset_view)],
    dataset_view_service: Annotated[DatasetViewService, Depends(get_dataset_view_service)],
) -> None:
    """
    Delete a dataset view.

    This operation only removes the view and its item assignments; it does NOT delete the underlying media
    or dataset items, which remain part of the project's main dataset.
    """
    dataset_view_service.delete_dataset_view(project_id=project.id, dataset_view_id=dataset_view.id)


@router.post(
    "/{dataset_view_id}/media",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        status.HTTP_204_NO_CONTENT: {"description": "Media successfully assigned to the dataset view"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid project, dataset view or media ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Project, dataset view or media not found"},
    },
)
def assign_media_to_dataset_view(
    project: Annotated[Project, Depends(get_project)],
    dataset_view: Annotated[DatasetView, Depends(get_dataset_view)],
    assign_media: Annotated[AssignMediaToDatasetView, Body()],
    dataset_view_service: Annotated[DatasetViewService, Depends(get_dataset_view_service)],
) -> None:
    """
    Assign one or more media items (in bulk) to a dataset view.

    The resulting content of the view can be listed with
    `GET /api/projects/{project_id}/dataset/media?dataset_view_id={dataset_view_id}`.
    """
    dataset_view_service.assign_media(
        project_id=project.id, dataset_view_id=dataset_view.id, media_ids=assign_media.media_ids
    )


@router.delete(
    "/{dataset_view_id}/media",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        status.HTTP_204_NO_CONTENT: {"description": "Media successfully unassigned from the dataset view"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid project, dataset view or media ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Project or dataset view not found"},
    },
)
def unassign_media_from_dataset_view(
    project: Annotated[Project, Depends(get_project)],
    dataset_view: Annotated[DatasetView, Depends(get_dataset_view)],
    unassign_media: Annotated[UnassignMediaFromDatasetView, Body()],
    dataset_view_service: Annotated[DatasetViewService, Depends(get_dataset_view_service)],
) -> None:
    """Unassign one or more media items (in bulk) from a dataset view. The media itself is not affected."""
    dataset_view_service.unassign_media(
        project_id=project.id, dataset_view_id=dataset_view.id, media_ids=unassign_media.media_ids
    )
