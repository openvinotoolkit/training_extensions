"""Tests whether the dataloaders can load the data correctly."""

# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from otx.api.entities.model_template import TaskType
from tests.unit.algorithms.anomaly.helpers.dummy_dataset import ShapesDataModule


@pytest.mark.parametrize(
    "task_type", [TaskType.ANOMALY_CLASSIFICATION, TaskType.ANOMALY_DETECTION, TaskType.ANOMALY_SEGMENTATION]
)
def test_dataloaders(task_type):
    """Tests whether the dataloaders can load the data correctly."""
    datamodule = ShapesDataModule(task_type)
    if task_type == TaskType.ANOMALY_CLASSIFICATION:
        assert next(iter(datamodule.predict_dataloader())).keys() == {"image", "label", "index"}
    elif task_type == TaskType.ANOMALY_DETECTION:
        assert next(iter(datamodule.predict_dataloader())).keys() == {"image", "label", "index", "boxes"}
    else:
        assert next(iter(datamodule.predict_dataloader())).keys() == {"image", "label", "index", "mask"}
