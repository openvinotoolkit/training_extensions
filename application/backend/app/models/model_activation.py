# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from app.models.system import DeviceInfo


class ModelActivationState(BaseModel):
    project_id: UUID | None = Field(..., description="Project ID of the model that is currently used for inference")
    active_model_id: UUID | None = Field(..., description="ID of the model that is currently used for inference")
    active_model_variant_id: UUID | None = Field(
        ..., description="ID of the model variant that is currently used for inference"
    )
    available_models: list[UUID] = Field(..., description="List of all available model IDs that can be activated")
    device: DeviceInfo = Field(..., description="Device information for inference")
    confidence_threshold: float | None = Field(
        default=None, description="Confidence threshold to apply to the model, or None to keep the model's own value"
    )
    label_colors: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of project label name to its hex colour, used to render predictions "
        "with the colours defined in the project",
    )

    @field_validator("active_model_id")
    @classmethod
    def validate_active_model(cls, v, info):  # noqa: ANN001
        if v is not None and "available_models" in info.data and v not in info.data["available_models"]:
            raise ValueError(
                f"active_model_id '{v}' must be one of the available_models: {info.data['available_models']}"
            )
        return v
