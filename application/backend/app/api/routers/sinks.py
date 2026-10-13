# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Endpoints for managing pipeline sinks"""

from typing import Annotated

import yaml
from fastapi import APIRouter, Body, Depends, File, UploadFile, status
from fastapi.exceptions import HTTPException
from fastapi.openapi.models import Example
from fastapi.responses import FileResponse, Response
from pydantic import ValidationError

from app.api.dependencies import get_sink, get_sink_service
from app.api.schemas.sink import SinkCreate, SinkCreateAdapter, SinkView, SinkViewAdapter
from app.api.schemas.test_result import TestResult
from app.models import Sink
from app.services import (
    ResourceInUseError,
    ResourceValidationError,
    ResourceWithIdAlreadyExistsError,
    ResourceWithNameAlreadyExistsError,
    SinkService,
)

router = APIRouter(prefix="/api/sinks", tags=["Sinks"])

CREATE_SINK_BODY_DESCRIPTION = """
Configuration for the new sink. The exact list of fields that can be configured depends on the sink type.
"""
CREATE_SINK_BODY_EXAMPLES = {
    "folder": Example(
        summary="Folder sink",
        description="Configuration for a local filesystem folder sink",
        value={
            "sink_type": "folder",
            "name": "My Output Folder",
            "folder_path": "/path/to/output",
            "output_formats": ["image_with_predictions"],
            "rate_limit": 0.2,
        },
    ),
    "mqtt": Example(
        summary="MQTT sink",
        description="Configuration for an MQTT message broker sink",
        value={
            "sink_type": "mqtt",
            "name": "Local MQTT Broker",
            "broker_host": "localhost",
            "broker_port": 1883,
            "topic": "predictions",
            "output_formats": ["predictions"],
        },
    ),
}

UPDATE_SINK_BODY_DESCRIPTION = """
Partial sink configuration update. May contain any subset of fields from the respective sink type
(e.g., 'broker_host' and 'broker_port' for MQTT; 'folder_path' for folder sinks).
Fields not included in the request will remain unchanged. The 'sink_type' field cannot be changed.
"""
UPDATE_SINK_BODY_EXAMPLES = {
    "folder": Example(
        summary="Update folder sink",
        description="Change the output path for a folder sink",
        value={
            "folder_path": "/new/output/directory",
        },
    ),
    "mqtt": Example(
        summary="Update MQTT sink",
        description="Change the topic for an MQTT sink",
        value={
            "topic": "new_predictions_topic",
        },
    ),
}


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    response_model=SinkView,
    responses={
        status.HTTP_201_CREATED: {"description": "Sink created"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid sink ID"},
        status.HTTP_409_CONFLICT: {"description": "Sink already exists"},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"description": "Sink validation failed"},
    },
)
def create_sink(
    sink_create: Annotated[
        SinkCreate, Body(description=CREATE_SINK_BODY_DESCRIPTION, openapi_examples=CREATE_SINK_BODY_EXAMPLES)
    ],
    sink_service: Annotated[SinkService, Depends(get_sink_service)],
) -> SinkView:
    """Create and configure a new sink.

    Returns 422 if the request body fails validation, or if the sink configuration
    is well-formed but not reachable (e.g. missing output folder, unreachable broker).
    """
    try:
        sink = sink_service.create_sink(
            name=sink_create.name,
            sink_type=sink_create.sink_type,
            rate_limit=sink_create.rate_limit,
            config_data=sink_create.config_data,
            output_formats=sink_create.output_formats,
            sink_id=sink_create.id,
        )
        return SinkViewAdapter.validate_python(sink, from_attributes=True)
    except (ResourceWithNameAlreadyExistsError, ResourceWithIdAlreadyExistsError) as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except ResourceValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(e))


@router.get(
    "",
    responses={
        status.HTTP_200_OK: {"description": "List of available sink configurations"},
    },
)
def list_sinks(
    sink_service: Annotated[SinkService, Depends(get_sink_service)],
) -> list[SinkView]:
    """List the available sinks"""
    sinks = sink_service.list_all()
    return [SinkViewAdapter.validate_python(sink, from_attributes=True) for sink in sinks]


@router.get(
    "/{sink_id}",
    responses={
        status.HTTP_200_OK: {"description": "Sink found"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid sink ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Sink not found"},
    },
)
def get_sink_view(sink: Annotated[Sink, Depends(get_sink)]) -> SinkView:
    """Get info about a sink"""
    return SinkViewAdapter.validate_python(sink, from_attributes=True)


@router.patch(
    "/{sink_id}",
    responses={
        status.HTTP_200_OK: {"description": "Sink successfully updated", "model": SinkView},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid sink ID or request body"},
        status.HTTP_404_NOT_FOUND: {"description": "Sink not found"},
        status.HTTP_409_CONFLICT: {"description": "Sink already exists"},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"description": "Sink validation failed"},
    },
)
def update_sink(
    sink: Annotated[Sink, Depends(get_sink)],
    sink_config: Annotated[
        dict,
        Body(
            description=UPDATE_SINK_BODY_DESCRIPTION,
            openapi_examples=UPDATE_SINK_BODY_EXAMPLES,
        ),
    ],
    sink_service: Annotated[SinkService, Depends(get_sink_service)],
) -> SinkView:
    """Reconfigure an existing sink.

    Returns 422 if the request body fails validation, or if the updated sink
    configuration is well-formed but not reachable (e.g. missing output folder,
    unreachable broker).
    """
    if "sink_type" in sink_config:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="The 'sink_type' field cannot be changed")
    try:
        updated_sink = sink.model_copy(update=sink_config)
        updated_sink.config_data = updated_sink.config_data.model_copy(  # pyrefly: ignore[bad-assignment]
            update=sink_config
        )
        sink = sink_service.update_sink(
            sink=sink,
            new_name=updated_sink.name,
            new_rate_limit=updated_sink.rate_limit,
            new_config_data=updated_sink.config_data,
            new_output_formats=updated_sink.output_formats,
        )
        return SinkViewAdapter.validate_python(sink, from_attributes=True)
    except ResourceWithNameAlreadyExistsError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except ResourceValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(e))


@router.post(
    "/{sink_id}:export",
    response_class=FileResponse,
    responses={
        status.HTTP_200_OK: {
            "description": "Sink configuration exported as a YAML file",
            "content": {
                "application/x-yaml": {"schema": {"type": "string", "format": "binary"}},
            },
        },
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid sink ID or request body"},
        status.HTTP_404_NOT_FOUND: {"description": "Sink not found"},
    },
)
def export_sink(sink: Annotated[Sink, Depends(get_sink)]) -> Response:
    """Export a sink to file"""
    yaml_content = yaml.safe_dump(sink.model_dump(mode="json", exclude={"id"}))

    return Response(
        content=yaml_content.encode("utf-8"),
        media_type="application/x-yaml",
        headers={"Content-Disposition": f"attachment; filename=sink_{sink.id}.yaml"},
    )


@router.post(
    ":import",
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_201_CREATED: {"description": "Sink imported successfully", "model": SinkView},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid YAML format"},
        status.HTTP_409_CONFLICT: {"description": "Sink already exists"},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"description": "Validation error(s)"},
    },
)
def import_sink(
    yaml_file: Annotated[UploadFile, File(description="YAML file containing the sink configuration")],
    sink_service: Annotated[SinkService, Depends(get_sink_service)],
) -> SinkView:
    """Import a sink from file"""
    try:
        yaml_content = yaml_file.file.read()
        sink_data = yaml.safe_load(yaml_content)
        sink_create = SinkCreateAdapter.validate_python(sink_data)
        sink = sink_service.create_sink(
            name=sink_create.name,
            sink_type=sink_create.sink_type,
            rate_limit=sink_create.rate_limit,
            config_data=sink_create.config_data,
            output_formats=sink_create.output_formats,
            sink_id=sink_create.id,
        )
        return SinkViewAdapter.validate_python(sink, from_attributes=True)
    except yaml.YAMLError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid YAML format: {str(e)}")
    except (ResourceWithNameAlreadyExistsError, ResourceWithIdAlreadyExistsError) as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(e))
    except ResourceValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(e))


@router.delete(
    "/{sink_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        status.HTTP_204_NO_CONTENT: {
            "description": "Sink configuration successfully deleted",
        },
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid sink ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Sink not found"},
        status.HTTP_409_CONFLICT: {"description": "Sink is used by at least one pipeline"},
    },
)
def delete_sink(
    sink: Annotated[Sink, Depends(get_sink)],
    sink_service: Annotated[SinkService, Depends(get_sink_service)],
) -> None:
    """Remove a sink"""
    try:
        sink_service.delete_sink(sink)
    except ResourceInUseError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))


@router.post(
    "/{sink_id}:test",
    response_model=TestResult,
    responses={
        status.HTTP_200_OK: {"description": "Sink connectivity test result", "model": TestResult},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid sink ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Sink not found"},
    },
)
def test_sink(
    sink: Annotated[Sink, Depends(get_sink)],
    sink_service: Annotated[SinkService, Depends(get_sink_service)],
) -> TestResult:
    """Test connectivity to a sink"""
    result = sink_service.test_sink(sink)
    return TestResult.model_validate(result, from_attributes=True)
