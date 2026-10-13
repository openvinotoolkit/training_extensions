# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Endpoints for managing pipeline sources"""

from typing import Annotated

import yaml
from fastapi import APIRouter, Body, Depends, File, UploadFile, status
from fastapi.exceptions import HTTPException
from fastapi.openapi.models import Example
from fastapi.responses import FileResponse, Response
from pydantic import ValidationError

from app.api.dependencies import get_source, get_source_update_service
from app.api.schemas.source import SourceCreate, SourceCreateAdapter, SourceView, SourceViewAdapter
from app.api.schemas.test_result import TestResult
from app.models import Source
from app.services import (
    ResourceInUseError,
    ResourceValidationError,
    ResourceWithIdAlreadyExistsError,
    ResourceWithNameAlreadyExistsError,
    SourceUpdateService,
)

router = APIRouter(prefix="/api/sources", tags=["Sources"])

CREATE_SOURCE_BODY_DESCRIPTION = """
Configuration for the new source. The exact list of fields that can be configured depends on the source type.
"""
CREATE_SOURCE_BODY_EXAMPLES = {
    "usb_camera": Example(
        summary="USB camera source",
        description="Configuration for a USB camera source",
        value={
            "source_type": "usb_camera",
            "name": "USB Camera 1",
            "device_id": 0,
        },
    ),
    "ip_camera": Example(
        summary="IP camera source",
        description="Configuration for an IP camera source",
        value={
            "source_type": "ip_camera",
            "name": "IP Camera 1",
            "stream_url": "rtsp://192.168.1.100:554/stream1",
            "auth_required": True,
        },
    ),
    "video_file": Example(
        summary="Video file source",
        description="Configuration for a video file source",
        value={
            "source_type": "video_file",
            "name": "Camera recording 123",
            "video_path": "/path/to/video.mp4",
            "loop": False,
        },
    ),
    "images_folder": Example(
        summary="Images folder source",
        description="Configuration for a folder containing images source",
        value={
            "source_type": "images_folder",
            "name": "Production Samples",
            "folder_path": "/path/to/images",
            "ignore_existing_images": True,
        },
    ),
}

UPDATE_SOURCE_BODY_DESCRIPTION = """
Partial source configuration update. May contain any subset of fields from the respective source type
(e.g., 'device_id' for USB camera; 'video_path' for video files).
Fields not included in the request will remain unchanged. The 'source_type' field cannot be changed.
"""
UPDATE_SOURCE_BODY_EXAMPLES = {
    "usb_camera": Example(
        summary="Update USB camera source",
        description="Rename a USB camera source",
        value={
            "name": "Updated USB camera name",
        },
    ),
    "video_file": Example(
        summary="Update video file source",
        description="Change the video path for a video file source",
        value={
            "video_path": "/new/path/to/video.mp4",
        },
    ),
}


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    response_model=SourceView,
    responses={
        status.HTTP_201_CREATED: {"description": "Source created"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid source ID or request body"},
        status.HTTP_409_CONFLICT: {"description": "Source already exists"},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"description": "Source validation failed"},
    },
)
def create_source(
    source_create: Annotated[
        SourceCreate, Body(description=CREATE_SOURCE_BODY_DESCRIPTION, openapi_examples=CREATE_SOURCE_BODY_EXAMPLES)
    ],
    source_update_service: Annotated[SourceUpdateService, Depends(get_source_update_service)],
) -> SourceView:
    """Create and configure a new source.

    Returns 422 if the request body fails validation, or if the source configuration
    is well-formed but not reachable (e.g. missing video file, unreachable camera).
    """
    try:
        source = source_update_service.create_source(
            name=source_create.name,
            source_type=source_create.source_type,
            config_data=source_create.config_data,
            source_id=source_create.id,
        )
        return SourceViewAdapter.validate_python(source, from_attributes=True)
    except (ResourceWithNameAlreadyExistsError, ResourceWithIdAlreadyExistsError) as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except ResourceValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(e))


@router.get(
    "",
    responses={
        status.HTTP_200_OK: {"description": "List of available source configurations"},
    },
)
def list_sources(
    source_update_service: Annotated[SourceUpdateService, Depends(get_source_update_service)],
) -> list[SourceView]:
    """List the available sources"""
    sources = source_update_service.list_all()
    return [SourceViewAdapter.validate_python(source, from_attributes=True) for source in sources]


@router.get(
    "/{source_id}",
    responses={
        status.HTTP_200_OK: {"description": "Source found"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid source ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Source not found"},
    },
)
def get_source_view(source: Annotated[Source, Depends(get_source)]) -> SourceView:
    """Get info about a source"""
    return SourceViewAdapter.validate_python(source, from_attributes=True)


@router.patch(
    "/{source_id}",
    responses={
        status.HTTP_200_OK: {"description": "Source successfully updated", "model": SourceView},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid source ID or request body"},
        status.HTTP_404_NOT_FOUND: {"description": "Source not found"},
        status.HTTP_409_CONFLICT: {"description": "Source already exists"},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"description": "Source validation failed"},
    },
)
def update_source(
    source: Annotated[Source, Depends(get_source)],
    source_config: Annotated[
        dict,
        Body(
            description=UPDATE_SOURCE_BODY_DESCRIPTION,
            openapi_examples=UPDATE_SOURCE_BODY_EXAMPLES,
        ),
    ],
    source_update_service: Annotated[SourceUpdateService, Depends(get_source_update_service)],
) -> SourceView:
    """Reconfigure an existing source.

    Returns 422 if the request body fails validation, or if the updated source
    configuration is well-formed but not reachable (e.g. missing video file,
    unreachable camera).
    """
    if "source_type" in source_config:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="The 'source_type' field cannot be changed")

    try:
        updated_source = source.model_copy(update=source_config)
        updated_source.config_data = updated_source.config_data.model_copy(  # pyrefly: ignore[bad-assignment]
            update=source_config
        )
        source = source_update_service.update_source(
            source=source,
            new_name=updated_source.name,
            new_config_data=updated_source.config_data,
        )
        return SourceViewAdapter.validate_python(source, from_attributes=True)
    except ResourceWithNameAlreadyExistsError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except ResourceValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(e))


@router.post(
    "/{source_id}:export",
    response_class=FileResponse,
    responses={
        status.HTTP_200_OK: {
            "description": "Source configuration exported as a YAML file",
            "content": {
                "application/x-yaml": {"schema": {"type": "string", "format": "binary"}},
            },
        },
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid source ID or request body"},
        status.HTTP_404_NOT_FOUND: {"description": "Source not found"},
    },
)
def export_source(source: Annotated[Source, Depends(get_source)]) -> Response:
    """Export a source to file"""
    yaml_content = yaml.safe_dump(source.model_dump(mode="json", exclude={"id"}))

    return Response(
        content=yaml_content.encode("utf8"),
        media_type="application/x-yaml",
        headers={"Content-Disposition": f"attachment; filename=source_{source.id}.yaml"},
    )


@router.post(
    ":import",
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_201_CREATED: {"description": "Source imported successfully", "model": SourceView},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid YAML format "},
        status.HTTP_409_CONFLICT: {"description": "Source already exists"},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"description": "Validation error(s)"},
    },
)
def import_source(
    yaml_file: Annotated[UploadFile, File(description="YAML file containing the source configuration")],
    source_update_service: Annotated[SourceUpdateService, Depends(get_source_update_service)],
) -> SourceView:
    """Import a source from file"""
    try:
        yaml_content = yaml_file.file.read()
        source_data = yaml.safe_load(yaml_content)
        source_create = SourceCreateAdapter.validate_python(source_data)
        source = source_update_service.create_source(
            name=source_create.name,
            source_type=source_create.source_type,
            config_data=source_create.config_data,
            source_id=source_create.id,
        )
        return SourceViewAdapter.validate_python(source, from_attributes=True)
    except yaml.YAMLError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid YAML format: {str(e)}")
    except (ResourceWithNameAlreadyExistsError, ResourceWithIdAlreadyExistsError) as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(e))
    except ResourceValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(e))


@router.delete(
    "/{source_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        status.HTTP_204_NO_CONTENT: {
            "description": "Source configuration successfully deleted",
        },
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid source ID or source is used by at least one pipeline"},
        status.HTTP_404_NOT_FOUND: {"description": "Source not found"},
        status.HTTP_409_CONFLICT: {"description": "Source is used by at least one pipeline"},
    },
)
def delete_source(
    source: Annotated[Source, Depends(get_source)],
    source_update_service: Annotated[SourceUpdateService, Depends(get_source_update_service)],
) -> None:
    """Remove a source"""
    try:
        source_update_service.delete_source(source)
    except ResourceInUseError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))


@router.post(
    "/{source_id}:test",
    response_model=TestResult,
    responses={
        status.HTTP_200_OK: {"description": "Source connectivity test result", "model": TestResult},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid source ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Source not found"},
    },
)
def test_source(
    source: Annotated[Source, Depends(get_source)],
    source_update_service: Annotated[SourceUpdateService, Depends(get_source_update_service)],
) -> TestResult:
    """Test connectivity to a source"""
    result = source_update_service.test_source(source)
    return TestResult.model_validate(result, from_attributes=True)
