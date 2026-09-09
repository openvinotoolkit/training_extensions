# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import StrEnum
from uuid import UUID

from loguru import logger
from pydantic import BaseModel, model_validator

from app.models import EvaluationResult


class EvaluationMetricName(StrEnum):
    ACCURACY = "Accuracy"
    PRECISION = "Precision"
    RECALL = "Recall"
    F_MEASURE = "F-measure"
    MAP = "mAP"  # mAP@0.5:0.95
    MAP_50 = "mAP@0.5"
    MAP_75 = "mAP@0.75"
    MAR_1 = "mAR@1"
    MAR_10 = "mAR@10"
    MAR_100 = "mAR@100"


# Mapping from raw metric names (returned by getitune) to user-friendly metric names (returned by the API).
# All keys in this mapping are lowercase; the mapping is applied in a case-insensitive manner.
RAW_METRICS_TO_API_METRICS = {
    "accuracy": EvaluationMetricName.ACCURACY,
    "precision": EvaluationMetricName.PRECISION,
    "recall": EvaluationMetricName.RECALL,
    "f_measure": EvaluationMetricName.F_MEASURE,
    "map": EvaluationMetricName.MAP,
    "map_50": EvaluationMetricName.MAP_50,
    "map_75": EvaluationMetricName.MAP_75,
    "mar_1": EvaluationMetricName.MAR_1,
    "mar_10": EvaluationMetricName.MAR_10,
    "mar_100": EvaluationMetricName.MAR_100,
}


class MetricView(BaseModel):
    """
    Represents a single evaluation metric with its name and value.

    Attributes:
        name: The metric name.
        value: The metric value as a floating-point number.
        primary: Indicates if this metric is the default choice for visualization purposes.
    """

    name: EvaluationMetricName
    value: float
    primary: bool = False

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "Accuracy",
                "value": 0.97,
                "primary": True,
            }
        }
    }


class EvaluationView(BaseModel):
    """
    Represents evaluation results for a dataset revision subset.

    This view provides a summary of model evaluation metrics obtained for a specific
    dataset revision and subset combination.

    Attributes:
        dataset_revision_id: Unique identifier of the evaluated dataset revision.
        subset: The dataset subset used for evaluation (e.g., "training", "validation", "testing").
        metrics: List of evaluation metrics with their corresponding values.
    """

    dataset_revision_id: UUID
    subset: str
    metrics: list[MetricView]

    model_config = {
        "json_schema_extra": {
            "example": {
                "dataset_revision_id": "3c6c6d38-1cd8-4458-b759-b9880c048b78",
                "subset": "testing",
                "metrics": [
                    {
                        "name": "Accuracy",
                        "value": 0.97,
                        "primary": True,
                    },
                    {
                        "name": "Precision",
                        "value": 0.98,
                        "primary": False,
                    },
                    {
                        "name": "Recall",
                        "value": 0.94,
                        "primary": False,
                    },
                ],
            }
        }
    }

    @classmethod
    def __serialize_metrics(cls, metrics: dict[str, float]) -> list[MetricView]:
        """
        Convert raw metric names and values into a list of MetricView instances with user-friendly names.

        This method maps raw metric names (e.g., "accuracy") to their corresponding
        user-friendly names defined in the EvaluationMetricName enum (e.g., "Accuracy").
        Only metrics that have an explicit mapping are included in the output and eventually returned by the API.

        Args:
            metrics: A dictionary of raw metric names and their float values.
        Returns:
            A list of MetricView instances with user-friendly metric names and their values.
        """
        raw_keys = {k.lower(): k for k in metrics}
        if "map_50" in raw_keys:
            primary_metric_name = EvaluationMetricName.MAP_50
        elif "map" in raw_keys:
            primary_metric_name = EvaluationMetricName.MAP
        elif "accuracy" in raw_keys:
            primary_metric_name = EvaluationMetricName.ACCURACY
        else:
            logger.error("Failed to resolve the primary evaluation metric. Raw metric keys: {}", raw_keys)
            raise ValueError("Unable to determine the primary evaluation metric.")

        metric_views = []
        for raw_name, value in metrics.items():
            api_metric_name = RAW_METRICS_TO_API_METRICS.get(raw_name.lower(), None)
            if api_metric_name is None:
                logger.debug(f"Skipping metric '{raw_name}' as it does not have a mapping to a user-friendly name.")
                continue
            primary = api_metric_name == primary_metric_name
            metric_views.append(MetricView(name=api_metric_name, value=value, primary=primary))
        return metric_views

    @model_validator(mode="before")
    @classmethod
    def populate_metrics(cls, data: object) -> object:
        if isinstance(data, EvaluationResult):
            return {
                "dataset_revision_id": data.dataset_revision_id,
                "subset": data.subset.value,
                "metrics": cls.__serialize_metrics(data.metrics),
            }
        if isinstance(data, dict):
            metrics = data["metrics"]
            if isinstance(metrics, dict):
                metrics = cls.__serialize_metrics(metrics)
            subset = data["subset"]
            return {
                "dataset_revision_id": data["dataset_revision_id"],
                "subset": subset.value if isinstance(subset, StrEnum) else subset,
                "metrics": metrics,
            }
        return data
