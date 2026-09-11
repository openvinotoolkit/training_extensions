# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from app.api.schemas.dataset import DatasetFilters
from app.core.jobs.models import JobType
from app.models import ExportDatasetJob

from .base import BaseJobRequest


class ExportDatasetParams(BaseModel):
    export_format: str = Field(..., description="The format of the dataset to export (e.g. coco)")
    filters: DatasetFilters = Field(default_factory=DatasetFilters, description="Dataset filters to use for export")


class ExportDatasetRequest(BaseJobRequest):
    job_type: Literal[JobType.EXPORT_DATASET]
    dataset_id: UUID | None = Field(
        None, description="The ID of the dataset revision used for export. If None, the project's main dataset is used."
    )
    dataset_view_id: UUID | None = Field(
        None, description="The ID of the dataset view to export. Mutually exclusive with dataset_id."
    )

    parameters: ExportDatasetParams = Field(..., description="The configuration for exporting dataset")

    @model_validator(mode="after")
    def validate_dataset_id_and_view_mutually_exclusive(self) -> "ExportDatasetRequest":
        """Validate that dataset_id and dataset_view_id are not both set."""
        if self.dataset_id is not None and self.dataset_view_id is not None:
            raise ValueError("dataset_id and dataset_view_id are mutually exclusive")
        return self


class StageDatasetParams(BaseModel):
    filters: DatasetFilters = Field(default_factory=DatasetFilters, description="Dataset filters to use for staging")


class StageDatasetRequest(BaseJobRequest):
    job_type: Literal[JobType.STAGE_DATASET]
    dataset_id: UUID | None = Field(
        None, description="The ID of the dataset used for staging. If None, the project's main dataset is used."
    )

    parameters: StageDatasetParams = Field(
        default_factory=StageDatasetParams, description="The configuration for staging dataset"
    )


class ExportDatasetMetadata(BaseModel):
    dataset_id: UUID | None = Field(None, description="Dataset ID")
    dataset_view_id: UUID | None = Field(None, description="Dataset view ID")
    project_id: UUID = Field(..., description="Project ID")
    filters: DatasetFilters = Field(..., description="Filters to apply to the dataset during export/staging")
    export_format: str | None = Field(None, description="The format of the dataset to export (e.g. coco)")

    @model_validator(mode="before")
    @classmethod
    def populate_metadata(cls, data: object) -> object:
        if isinstance(data, ExportDatasetJob):
            return {
                "dataset_id": data.params.dataset_id,
                "dataset_view_id": data.params.dataset_view_id,
                "project_id": data.project_id,
                "filters": DatasetFilters(
                    labels=data.params.labels,
                    subsets=data.params.subsets,
                    include_unannotated=data.params.include_unannotated,
                ),
                "export_format": data.params.export_format.value,
            }
        return data
