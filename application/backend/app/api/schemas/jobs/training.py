# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Literal, Self
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from app.api.schemas.system import DeviceInfoView
from app.core.jobs.models import JobType
from app.models import TrainingJob

from .base import BaseJobRequest


class TrainingRequestParams(BaseModel):
    """Request parameters for the training job"""

    device: str = Field(..., description="Device identifier for training (e.g., 'cpu', 'xpu-0', 'cuda-1')")
    model_architecture_id: str = Field(..., description="Model architecture identifier")
    parent_model_revision_id: UUID | None = Field(
        None, description="Parent model revision ID for fine-tuning, null for training from scratch"
    )
    dataset_revision_id: UUID | None = Field(
        None,
        description="Dataset revision ID if reusing an existing dataset revision, "
        "null if training on the latest dataset",
    )
    dataset_view_id: UUID | None = Field(
        None,
        description="Dataset view ID to train only on the media assigned to that view; "
        "mutually exclusive with dataset_revision_id, null if training on the entire dataset",
    )

    @model_validator(mode="after")
    def validate_dataset_selection(self) -> Self:
        if self.dataset_revision_id is not None and self.dataset_view_id is not None:
            raise ValueError("Parameters 'dataset_revision_id' and 'dataset_view_id' are mutually exclusive")
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "device": "xpu-0",
                "model_architecture_id": "object-detection-atss-mobilenet-v2",
                "parent_model_revision_id": "ef3983f1-cef0-4ebe-91db-7330f1dd6e27",
                "dataset_revision_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                "dataset_view_id": None,
            }
        }
    }


class TrainingRequest(BaseJobRequest):
    """Request schema for training a new model."""

    job_type: Literal[JobType.TRAIN]
    parameters: TrainingRequestParams = Field(..., description="Parameters required for the training job")

    model_config = {
        "json_schema_extra": {
            "example": {
                "job_type": "train",
                "project_id": "7b073838-99d3-42ff-9018-4e901eb047fc",
                "parameters": {
                    "device": "xpu-0",
                    "model_architecture_id": "object-detection-atss-mobilenet-v2",
                    "parent_model_revision_id": "ef3983f1-cef0-4ebe-91db-7330f1dd6e27",
                    "dataset_revision_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                    "dataset_view_id": None,
                },
            }
        }
    }


class ProjectMetadata(BaseModel):
    """Metadata about a project."""

    id: UUID = Field(..., description="Project identifier")


class ModelMetadata(BaseModel):
    """Metadata about a model."""

    id: UUID = Field(..., description="Model identifier")
    name: str = Field(..., description="Model name")
    architecture: str = Field(..., description="Model architecture identifier")
    parent_revision_id: UUID | None = Field(
        None, description="Parent model revision ID for fine-tuning, null if trained from scratch"
    )
    dataset_revision_id: UUID | None = Field(
        None,
        description="Dataset revision ID if reusing an existing dataset revision, "
        "null if training on the latest dataset",
    )
    dataset_view_id: UUID | None = Field(
        None, description="Dataset view the model is being trained on, null if training on the entire dataset"
    )
    dataset_view_name: str | None = Field(
        None,
        description="Name of the dataset view the model is being trained on, null if training on the entire dataset",
    )


class TrainingMetadata(BaseModel):
    """Metadata associated with a training job."""

    project: ProjectMetadata = Field(..., description="Project associated with the training job")
    model: ModelMetadata = Field(..., description="Model being trained")
    device: DeviceInfoView = Field(..., description="Device associated with the training job")

    @model_validator(mode="before")
    @classmethod
    def populate_metadata(cls, data: object) -> object:
        if isinstance(data, TrainingJob):
            return {
                "project": ProjectMetadata(id=data.project_id),
                "model": ModelMetadata(
                    id=data.params.model_id,
                    name=data.params.model_name,
                    architecture=data.params.model_architecture_id,
                    parent_revision_id=data.params.parent_model_revision_id,
                    dataset_revision_id=data.params.dataset_revision_id,
                    dataset_view_id=data.params.dataset_view_id,
                    dataset_view_name=data.params.dataset_view_name,
                ),
                "device": DeviceInfoView.model_validate(data.params.device, from_attributes=True),
            }
        return data
