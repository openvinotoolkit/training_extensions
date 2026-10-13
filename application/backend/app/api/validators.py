# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from datetime import UTC, datetime
from typing import Annotated
from uuid import UUID

from fastapi import HTTPException
from pydantic import BeforeValidator
from starlette import status


def validate_uuid_param(value: str, param_name: str) -> UUID:
    """
    Validate and convert a string to a UUID object.

    This function validates that the provided string value is a valid UUID format
    and converts it to a UUID object. If validation fails, it raises an HTTPException
    with a descriptive error message.

    Args:
        value: The string value to validate as a UUID.
        param_name: The name of the parameter for error messaging. Defaults to "ID".

    Returns:
        A UUID object if validation is successful.

    Raises:
        HTTPException: A 400 Bad Request error if the value is not a valid UUID format.

    Examples:
        >>> validate_uuid_param("550e8400-e29b-41d4-a716-446655440000", "user_id")
        UUID('550e8400-e29b-41d4-a716-446655440000')

        >>> validate_uuid_param("invalid-uuid", "user_id")
        HTTPException: Invalid user_id: 'invalid-uuid' must be a valid UUID format

    """
    try:
        return UUID(value)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid {param_name}: '{value}' must be a valid UUID format.",
        )


def normalize_datetime_to_utc(value: datetime | None) -> datetime | None:
    """Normalize datetimes to UTC and attach UTC tzinfo for naive values."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _uuid_path_validator(param_name: str) -> BeforeValidator:
    def validate(value: str) -> UUID:
        return validate_uuid_param(value, param_name)

    return BeforeValidator(validate)


JobID = Annotated[UUID, _uuid_path_validator("job_id")]
SourceID = Annotated[UUID, _uuid_path_validator("source_id")]
SinkID = Annotated[UUID, _uuid_path_validator("sink_id")]
ProjectID = Annotated[UUID, _uuid_path_validator("project_id")]
MediaID = Annotated[UUID, _uuid_path_validator("media_id")]
ModelID = Annotated[UUID, _uuid_path_validator("model_id")]
ModelVariantID = Annotated[UUID, _uuid_path_validator("model_variant_id")]
DatasetItemID = Annotated[UUID, _uuid_path_validator("dataset_item_id")]
DatasetRevisionID = Annotated[UUID, _uuid_path_validator("dataset_revision_id")]
DatasetViewID = Annotated[UUID, _uuid_path_validator("dataset_view_id")]
StagedDatasetID = Annotated[UUID, _uuid_path_validator("staged_dataset_id")]
