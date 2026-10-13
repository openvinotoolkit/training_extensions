# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from datetime import datetime
from uuid import UUID

from pydantic import model_validator

from app.db.schema import DatasetRevisionDB
from app.models.base import BaseEntity


class DatasetRevisionCounts(BaseEntity):
    """
    Counts of samples in different splits of a dataset revision.
    """

    total: int
    training: int
    validation: int
    testing: int

    @model_validator(mode="before")
    @classmethod
    def populate_training_info(cls, data: object) -> object:
        if isinstance(data, DatasetRevisionDB):
            return {
                "total": data.total_count,
                "training": data.training_count,
                "validation": data.validation_count,
                "testing": data.testing_count,
            }
        return data


class DatasetRevision(BaseEntity):
    """
    A dataset revision is an immutable snapshot of a training dataset.

    Attributes:
        id: Unique identifier for the dataset revision.
        name: Name of the dataset revision.
        created_at: Timestamp indicating when the dataset revision was created.
        files_deleted: Flag indicating whether the files associated with this dataset revision have been deleted.
        size: Size on disk in bytes
        item_counts: Number of items in the subsets Training, Validation and Testing and total count.
        dataset_view_id: Identifier of the dataset view the revision was created from, or None if it covers
            the entire dataset.
    """

    id: UUID
    name: str
    created_at: datetime
    files_deleted: bool
    size: int
    item_counts: DatasetRevisionCounts
    dataset_view_id: UUID | None = None

    @model_validator(mode="before")
    @classmethod
    def populate_model_revision(cls, data: object) -> object:
        if isinstance(data, DatasetRevisionDB):
            return {
                "id": data.id,
                "name": data.name,
                "created_at": data.created_at,
                "files_deleted": data.files_deleted,
                "size": data.size,
                "item_counts": DatasetRevisionCounts.model_validate(data),
                "dataset_view_id": data.dataset_view_id,
            }
        return data
