# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import io
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, cast

from fastapi import APIRouter, Body, Depends, Header, HTTPException, Query, status
from fastapi.openapi.models import Example
from fastapi.responses import StreamingResponse
from starlette.responses import FileResponse

from app.api.dependencies import (
    get_demo_files_service,
    get_inference_server,
    get_model_service,
    get_project,
    get_training_configuration_service,
)
from app.api.schemas import ModelView, ProjectView, TrainingConfigurationView, TrainingMetricsView
from app.api.validators import DatasetRevisionID, ModelID, ModelVariantID
from app.services import (
    ModelService,
    ResourceInUseError,
    ResourceNotFoundError,
    ResourceType,
    TrainingConfigurationService,
)
from app.services.demo_files_service import DemoFilesService
from app.services.inference import InferenceServer

router = APIRouter(prefix="/api/projects/{project_id}/models", tags=["Models"])


@router.get(
    "",
    response_model=list[ModelView],
    responses={
        status.HTTP_200_OK: {"description": "List of available models"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid project ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Project not found"},
    },
)
def list_models(
    project: Annotated[ProjectView, Depends(get_project)],
    model_service: Annotated[ModelService, Depends(get_model_service)],
    dataset_revision_id: Annotated[
        DatasetRevisionID | None,
        Query(description="Dataset revision id for optional filtering"),
    ] = None,
) -> list[ModelView]:
    """Get all models in a project, optionally filtered by dataset revision."""
    model_views = []
    for model_revision in model_service.list_models(project_id=project.id, dataset_revision_id=dataset_revision_id):
        model_variants = model_service.get_model_variants(project_id=project.id, model_id=model_revision.id)
        model_size = model_service.get_model_size_in_bytes(project_id=project.id, model_id=model_revision.id)
        model_views.append(model_revision.model_dump() | {"variants": model_variants} | {"size": model_size})
    return [ModelView.model_validate(model_view, from_attributes=True) for model_view in model_views]


@router.get(
    "/{model_id}",
    response_model=ModelView,
    responses={
        status.HTTP_200_OK: {"description": "Model found"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid project or model ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Project or model not found"},
    },
)
def get_model(
    project: Annotated[ProjectView, Depends(get_project)],
    model_id: ModelID,
    model_service: Annotated[ModelService, Depends(get_model_service)],
) -> ModelView:
    """Get a specific model by ID."""
    model_revision = model_service.get_model(project_id=project.id, model_id=model_id)
    model_variants = model_service.get_model_variants(project_id=project.id, model_id=model_id)
    model_size = model_service.get_model_size_in_bytes(project_id=project.id, model_id=model_id)
    model_view = model_revision.model_dump() | {"variants": model_variants} | {"size": model_size}
    return ModelView.model_validate(model_view, from_attributes=True)


@router.get(
    "/{model_id}/variants/{model_variant_id}/binary",
    responses={
        status.HTTP_200_OK: {"description": "Model weights of the model variant (zip archive)"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid project or model ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Project or model not found"},
    },
)
def download_model_binary(
    project: Annotated[ProjectView, Depends(get_project)],
    model_service: Annotated[ModelService, Depends(get_model_service)],
    demo_files_service: Annotated[DemoFilesService, Depends(get_demo_files_service)],
    model_id: ModelID,
    model_variant_id: ModelVariantID,
) -> StreamingResponse:
    """Download trained model weights of a desired model variant as a zip archive.

    For deployable formats (OpenVINO and ONNX), the archive also includes: a sample
    image taken from the project's dataset, ready-to-run sync/async inference
    demo scripts, a requirements.txt, and a README.md.
    """
    files_exist, paths = model_service.get_model_binary_files(
        project_id=project.id,
        model_id=model_id,
        model_variant_id=model_variant_id,
    )
    if not files_exist:
        raise ResourceNotFoundError(
            resource_type=ResourceType.MODEL,
            resource_id=f"{model_id} with variant {model_variant_id}",
        )

    model_variant = model_service.get_variant(variant_id=model_variant_id)
    filename = f"model-{str(model_id).split('-')[0]}-{model_variant.format}-{model_variant.precision}.zip"

    license = model_service.get_model_license(project_id=project.id, model_id=model_id)
    attribution = model_service.get_model_attribution(project_id=project.id, model_id=model_id)

    demo_files = demo_files_service.build_demo_files(
        project_id=project.id,
        model_format=model_variant.format,
        license=license,
        attribution=attribution,
    )

    # Create an in-memory zip file
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for path in paths:
            variant_dir = next(parent for parent in path.parents if parent.name == str(model_variant_id))
            zip_file.write(path, arcname=path.relative_to(variant_dir))
        for demo_file in demo_files:
            zip_file.writestr(demo_file.name, demo_file.data)

    zip_buffer.seek(0)

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


UPDATE_MODEL_BODY_DESCRIPTION = """
Update name of model revision.
"""
UPDATE_MODEL_BODY_EXAMPLES = {
    "name": Example(
        summary="Update model name",
        description="Change the name of the model",
        value={
            "name": "new_model_name",
        },
    ),
}


@router.patch(
    "/{model_id}",
    response_model=ModelView,
    responses={
        status.HTTP_200_OK: {"description": "Model successfully renamed"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid project or model ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Project or model not found"},
    },
)
def rename_model(
    project: Annotated[ProjectView, Depends(get_project)],
    model_id: ModelID,
    model_metadata: Annotated[
        dict,
        Body(
            description=UPDATE_MODEL_BODY_DESCRIPTION,
            openapi_examples=UPDATE_MODEL_BODY_EXAMPLES,
        ),
    ],
    model_service: Annotated[ModelService, Depends(get_model_service)],
) -> ModelView:
    """Rename a model"""
    model_revision = model_service.rename_model(project_id=project.id, model_id=model_id, model_metadata=model_metadata)
    model_variants = model_service.get_model_variants(project_id=project.id, model_id=model_id)
    model_size = model_service.get_model_size_in_bytes(project_id=project.id, model_id=model_id)
    model_view = model_revision.model_dump() | {"variants": model_variants} | {"size": model_size}
    return ModelView.model_validate(model_view, from_attributes=True)


@router.delete(
    "/{model_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        status.HTTP_204_NO_CONTENT: {
            "description": "Model configuration successfully deleted",
        },
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid project or model ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Project or model not found"},
        status.HTTP_409_CONFLICT: {"description": "Model is used by at least one pipeline"},
    },
)
def delete_model(
    project: Annotated[ProjectView, Depends(get_project)],
    model_id: ModelID,
    model_service: Annotated[ModelService, Depends(get_model_service)],
    inference_server: Annotated[InferenceServer, Depends(get_inference_server)],
    files_only: Annotated[bool, Query()] = False,
) -> None:
    """
    Delete a given model, or the files associated with it.

    If `files_only` is false, the entire model revision is deleted.
    If `files_only` is true, only the model files are deleted (weights, training logs, ...), while model metadata
    such as name and creation date are preserved; the model continues to exist as a lightweight record in the system,
    although operations that depend on the presence of model files (e.g., fine-tuning, inference) will no longer be
    possible; the only purpose of this option is to free up storage space while preserving model metadata.
    """
    try:
        if files_only:
            model_service.delete_model_files(project_id=project.id, model_id=model_id)
        else:
            model_service.delete_model(project_id=project.id, model_id=model_id)
        inference_server.invalidate_model(project_id=project.id, model_id=model_id)
    except ResourceInUseError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))


@router.get(
    "/{model_id}/training_configuration",
    response_model=TrainingConfigurationView,
    responses={
        status.HTTP_200_OK: {"description": "Training configuration for the model"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid project or model ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Project or model not found"},
    },
)
def get_model_training_configuration(
    project: Annotated[ProjectView, Depends(get_project)],
    model_id: ModelID,
    model_service: Annotated[ModelService, Depends(get_model_service)],
    training_configuration_service: Annotated[
        TrainingConfigurationService, Depends(get_training_configuration_service)
    ],
) -> TrainingConfigurationView:
    """
    Get the configuration used to train a given model, including both the task-level and algorithm-level parameters.
    """
    training_configuration = training_configuration_service.get_by_model_revision(
        project_id=project.id,
        model_revision_id=model_id,
    )
    model_architecture_id = model_service.get_model_revision_architecture(project_id=project.id, model_id=model_id)
    default_config = TrainingConfigurationService.get_default_by_model_architecture(
        model_architecture_id=model_architecture_id
    )
    return TrainingConfigurationView.from_training_configuration(
        config=training_configuration, default_config=default_config
    )


@router.get(
    "/{model_id}/training_metrics",
    response_model=TrainingMetricsView,
    responses={
        status.HTTP_200_OK: {"description": "Training metrics for the model"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid project or model ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Project, model, or metrics file not found"},
    },
)
def get_training_metrics(
    project: Annotated[ProjectView, Depends(get_project)],
    model_id: ModelID,
    model_service: Annotated[ModelService, Depends(get_model_service)],
) -> TrainingMetricsView:
    """
    Get metrics computed at training time, such as loss over time, validation accuracy over time, etc.
    """
    training_metrics = model_service.get_model_training_metrics(project_id=project.id, model_id=model_id)
    return TrainingMetricsView.model_validate({"training_metrics": training_metrics})


@router.get(
    "/{model_id}/logs",
    response_model=None,
    responses={
        status.HTTP_200_OK: {
            "description": "Training logs for the model",
            "content": {
                "application/x-ndjson": {"description": "NDJSON format (newline-delimited JSON)"},
                "text/plain": {"description": "Plain text format (extracted from NDJSON)"},
            },
        },
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid project or model ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Project, model or log file not found"},
        status.HTTP_409_CONFLICT: {
            "description": "Logs cannot be retrieved for models in not started or in-progress state"
        },
    },
)
def get_training_logs(
    project: Annotated[ProjectView, Depends(get_project)],
    model_id: ModelID,
    model_service: Annotated[ModelService, Depends(get_model_service)],
    accept: Annotated[str | None, Header(description="Accept header to specify the desired response format")] = None,
) -> FileResponse | StreamingResponse:
    """
    Download the training log file for a given model.
    Supports content negotiation via Accept header (text/plain, application/x-ndjson).
    """
    try:
        as_text = accept is not None and "text/plain" in accept.lower()

        training_log = model_service.get_logs(project_id=project.id, model_id=model_id, as_text=as_text)
        if training_log is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Log file not found")

        if as_text:
            return StreamingResponse(
                content=cast(Iterable, training_log),
                media_type="text/plain",
                headers={"Content-Disposition": f'attachment; filename="training-{model_id}.log"'},
            )

        log_file = cast(Path, training_log)
        return FileResponse(log_file, media_type="application/x-ndjson", filename=log_file.name)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
