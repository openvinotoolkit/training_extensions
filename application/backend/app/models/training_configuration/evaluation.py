# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import StrEnum

from pydantic import BaseModel, Field


class ValidationMetric(StrEnum):
    """Available metrics for model evaluation during validation."""

    # TODO (#5521): Enable additional metrics
    DEFAULT = "default"  # Auto-select the most appropriate metric based on the task type


class TaskLevelEvaluationParameters(BaseModel):
    """Evaluation parameters that apply at the task level and are relevant for all models architectures."""

    validation_metric: ValidationMetric = Field(
        default=ValidationMetric.DEFAULT,
        title="Validation metric",
        description="Metric used to evaluate model performance during validation.",
    )
