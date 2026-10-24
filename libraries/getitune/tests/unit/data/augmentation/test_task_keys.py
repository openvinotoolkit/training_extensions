# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the shared GPU-augmentation task-keys table."""

from __future__ import annotations

from getitune.data.augmentation.task_keys import DATA_KEYS_BY_TASK
from getitune.types.task import TaskType


def test_every_task_type_has_an_entry() -> None:
    for task in TaskType:
        assert task in DATA_KEYS_BY_TASK


def test_detection_transforms_boxes_and_labels() -> None:
    assert DATA_KEYS_BY_TASK[TaskType.DETECTION] == ("bbox_xyxy", "label")


def test_instance_segmentation_transforms_boxes_masks_and_labels() -> None:
    assert DATA_KEYS_BY_TASK[TaskType.INSTANCE_SEGMENTATION] == ("bbox_xyxy", "mask", "label")


def test_semantic_segmentation_transforms_only_masks() -> None:
    assert DATA_KEYS_BY_TASK[TaskType.SEMANTIC_SEGMENTATION] == ("mask",)


def test_classification_tasks_transform_only_labels() -> None:
    assert DATA_KEYS_BY_TASK[TaskType.MULTI_CLASS_CLS] == ("label",)
    assert DATA_KEYS_BY_TASK[TaskType.MULTI_LABEL_CLS] == ("label",)
