# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from getitune import TaskType as GetiTuneTaskType
    from getitune.config.data import SubsetConfig
    from getitune.data.dataset.base import VisionDataset
    from getitune.metrics import MetricCallable


from app.models import Task, TaskType


def get_getitune_task_type_by_task(task: Task) -> GetiTuneTaskType:
    """Map internal Task to GetiTuneTaskType."""
    from getitune import TaskType as GetiTuneTaskType

    match task.task_type:
        case TaskType.CLASSIFICATION:
            if task.exclusive_labels:
                return GetiTuneTaskType.MULTI_CLASS_CLS
            return GetiTuneTaskType.MULTI_LABEL_CLS
        case TaskType.DETECTION:
            return GetiTuneTaskType.DETECTION
        case TaskType.INSTANCE_SEGMENTATION:
            return GetiTuneTaskType.INSTANCE_SEGMENTATION
        case _:
            raise ValueError(f"Unsupported task type: {task.task_type}")


def get_metric_by_task(task: Task) -> MetricCallable:
    """Map internal Task to Metric."""
    from getitune.metrics.accuracy import MultiClassClsMetricCallable, MultiLabelClsMetricCallable
    from getitune.metrics.mean_ap import MaskRLEMeanAPCallable, MeanAPCallable

    match task.task_type:
        case TaskType.CLASSIFICATION:
            if task.exclusive_labels:
                return MultiClassClsMetricCallable
            return MultiLabelClsMetricCallable
        case TaskType.DETECTION:
            return MeanAPCallable
        case TaskType.INSTANCE_SEGMENTATION:
            return MaskRLEMeanAPCallable
        case _:
            raise ValueError(f"Unsupported task type: {task.task_type}")


def get_getitune_dataset_class_by_task_type(getitune_task_type: GetiTuneTaskType) -> type[VisionDataset]:
    """Get the VisionDataset class corresponding to the given GetiTuneTaskType."""
    from getitune import TaskType as GetiTuneTaskType
    from getitune.data import DetectionDataset, InstanceSegDataset, MulticlassClsDataset, MultilabelClsDataset

    task_type_to_class: dict[GetiTuneTaskType, type[VisionDataset]] = {
        GetiTuneTaskType.MULTI_CLASS_CLS: MulticlassClsDataset,
        GetiTuneTaskType.MULTI_LABEL_CLS: MultilabelClsDataset,
        GetiTuneTaskType.DETECTION: DetectionDataset,
        GetiTuneTaskType.INSTANCE_SEGMENTATION: InstanceSegDataset,
    }
    try:
        return task_type_to_class[getitune_task_type]
    except KeyError:
        raise ValueError(f"Unsupported getitune task type: {getitune_task_type}")


def build_getitune_subset_config(getitune_training_config: dict, subset_name: str) -> SubsetConfig:
    """Build the getitune SubsetConfig (data loader parameters and transforms) for a subset.

    Args:
        getitune_training_config: Training configuration in the format consumed by getitune.
        subset_name: Name of the subset, one of 'train', 'val' or 'test'.

    Returns:
        The SubsetConfig with its transforms already instantiated.
    """
    from getitune.config.data import SamplerConfig, SubsetConfig
    from getitune.data.factory import TransformLibFactory

    subset_cfg_data = getitune_training_config["data"][f"{subset_name}_subset"]
    subset_cfg_data["input_size"] = getitune_training_config["data"]["input_size"]
    sampler_cfg_data = subset_cfg_data.pop("sampler", {})
    subset_config = SubsetConfig(sampler=SamplerConfig(**sampler_cfg_data), **subset_cfg_data)
    # pyrefly: ignore[missing-attribute,bad-assignment]
    subset_config.transforms = TransformLibFactory.generate(subset_config)
    return subset_config


def convert_metrics(metrics: dict) -> dict[str, float]:
    """Convert metric values to a flat dict of scalar floats.

    Handles torch.Tensor values that may contain multiple elements
    (e.g., per-class metrics) by skipping them, matching the behavior
    of OVEngine.log_results.
    """
    import torch

    result: dict[str, float] = {}
    for k, v in metrics.items():
        name = k.split("/", 1)[1] if "/" in k else k
        if isinstance(v, torch.Tensor):
            if v.numel() == 1:
                result[name] = v.item()
            else:
                logger.debug("Skipping non-scalar metric '{}' with {} elements", name, v.numel())
        elif isinstance(v, list | tuple | dict) or (isinstance(v, np.ndarray) and v.size != 1):
            # Skip non-scalar aggregate metrics (e.g., per-class confusion matrices)
            logger.debug("Skipping non-scalar metric '{}' of type {}", name, type(v).__name__)
        elif isinstance(v, np.ndarray):
            result[name] = float(v.item())
        else:
            try:
                result[name] = float(v)
            except (TypeError, ValueError):
                logger.debug("Skipping metric '{}' with non-numeric value of type {}", name, type(v).__name__)
    return result
