# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from typing import Generic, TypeVar

from pydantic import BaseModel, Field, model_validator

from app.core.models import HasID, RequiresID
from app.models import TaskType

from .label import LabelCreate, LabelView


class TaskBase(BaseModel):
    task_type: TaskType = Field(description="Task type (classification, detection or instance_segmentation).")
    exclusive_labels: bool = Field(
        default=False,
        description="Whether labels are exclusive (multi class classification) or not (multi label classification).",
    )


class TaskView(TaskBase):
    labels: list[LabelView] = Field(default_factory=list, description="List of task labels.")


class TaskCreate(TaskBase):
    labels: list[LabelCreate] = Field(default_factory=list, description="List of task labels to create.")

    @model_validator(mode="after")
    def validate_labels(self) -> "TaskCreate":
        if self.task_type is TaskType.CLASSIFICATION and self.exclusive_labels and len(self.labels) < 2:
            raise ValueError("Multi-class classification requires at least two labels.")
        if len(self.labels) == 0:
            raise ValueError("A project requires at least one label.")
        return self


class ProjectUpdateName(BaseModel):
    """Schema for updating project name"""

    name: str

    model_config = {"json_schema_extra": {"example": {"name": "updated_project_name"}}}


T = TypeVar("T", bound=TaskBase)  # Task type e.g. TaskView, TaskCreate


class ProjectBase(BaseModel, Generic[T]):
    name: str = Field(..., description="Name of the project.")
    task: T

    @classmethod
    def example_dict(cls, view: bool = False) -> dict:
        """Generate example with configurable ID inclusion."""
        labels = [
            {
                **({"id": "a22d82ba-afa9-4d6e-bbc1-8c8e4002ec29", "color": "#FF5733", "hotkey": "S"} if view else {}),
                "name": "cat",
            },
            {
                **({"id": "8aa85368-11ba-4507-88f2-6a6704d78ef5", "color": "#33FF57", "hotkey": "D"} if view else {}),
                "name": "dog",
            },
        ]

        return {
            **({"id": "7b073838-99d3-42ff-9018-4e901eb047fc"} if view else {}),
            "name": "animals",
            "active_pipeline": False if view else None,
            **({"created_at": "2025-01-01T00:00:00Z"} if view else {}),
            "task": {
                "task_type": "classification",
                "exclusive_labels": True,
                "labels": labels,
            },
        }


class ProjectCreate(HasID, ProjectBase[TaskCreate]):
    model_config = {
        "json_schema_extra": {
            "example": ProjectBase.example_dict(),
        }
    }


class ModelCountsView(BaseModel):
    trained_models: int = Field(..., ge=0, description="Number of models whose training completed successfully.")


class ProjectView(RequiresID, ProjectBase[TaskView]):
    active_pipeline: bool = Field(..., description="Whether the project has an active pipeline.")
    created_at: datetime = Field(..., description="Timestamp when the project was created.")
    model_counts: ModelCountsView | None = Field(
        default=None,
        description="Model counts of the project. Only populated when requested with `include_model_counts=true`.",
    )

    model_config = {
        "json_schema_extra": {
            "example": ProjectBase.example_dict(view=True),
        }
    }
