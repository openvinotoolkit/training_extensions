# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID

import numpy as np
from pydantic import BaseModel, Field

from app.models import DatasetItemAnnotation
from app.models.system import DeviceInfo


@dataclass(frozen=True)
class BatchInferenceInput:
    media_id: UUID
    data: np.ndarray
    frame_index: int | None = None


@dataclass(frozen=True)
class BatchInferenceMedia:
    id: UUID
    frame_index: int | None = None


@dataclass(frozen=True)
class BatchInferencePrediction:
    media: BatchInferenceMedia
    prediction: list[DatasetItemAnnotation]


@dataclass(frozen=True)
class BatchInferenceResult:
    predictions: list[BatchInferencePrediction]


@dataclass(frozen=True)
class InferenceModel:
    """
    Information about model used for inference it's load time and device.
    """

    model_id: UUID
    device: DeviceInfo
    load_timestamp: datetime


class InferenceStatus(StrEnum):
    """
    Inference server status. Can be idle, loading or active
    """

    IDLE = "IDLE"
    LOADING = "LOADING"
    ACTIVE = "ACTIVE"


@dataclass(frozen=True)
class InferenceState:
    """
    Inference server status.
    Contains information about the current inference state and model if loaded.
    """

    status: InferenceStatus
    models: tuple[InferenceModel, ...] = ()


class InferenceWorkerStatusCode(StrEnum):
    """Status codes reported by the InferenceWorker process via IPC."""

    OK = "ok"
    ERROR = "error"


class InferenceWorkerStatus(BaseModel):
    """
    Status report emitted by InferenceWorker to communicate inference state
    to the Scheduler (or any consumer in the main process) via shared memory.

    Attributes:
        code: High-level status category.
        model_id: ID of the model being used.
        message: Optional free-form description or error message.
        timestamp: When the status was generated.
    """

    code: InferenceWorkerStatusCode
    model_id: UUID
    message: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def is_error(self) -> bool:
        return self.code == InferenceWorkerStatusCode.ERROR
