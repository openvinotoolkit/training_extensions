"""Tests whether the dataloaders can load the data correctly."""

# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from otx.api.entities.model_template import TaskType
from tests.unit.algorithms.anomaly.helpers.dummy_dataset import HazelnutDataModule


@pytest.mark.parametrize("stage", ["predict", "fit", "validate", "test"])
@pytest.mark.parametrize("task_type", [TaskType.ANOMALY_CLASSIFICATION, TaskType.ANOMALY_DETECTION])
def test_dataloaders(task_type, stage):
    """Tests whether the datamodule can load the data correctly.

    For all the test stages and the task types, the datamodule should return the correct keys.
    """
    datamodule = HazelnutDataModule(task_type)
    datamodule.setup(stage)
    if stage == "fit":
        batch = next(iter(datamodule.train_dataloader()))
    elif stage == "validate":
        batch = next(iter(datamodule.val_dataloader()))
    elif stage == "test":
        batch = next(iter(datamodule.test_dataloader()))
    else:
        batch = next(iter(datamodule.predict_dataloader()))
    if task_type == TaskType.ANOMALY_CLASSIFICATION:
        assert batch.keys() == {"image", "label", "index"}
    elif task_type == TaskType.ANOMALY_DETECTION:
        assert batch.keys() == {"image", "label", "index", "boxes"}
    else:
        assert batch.keys() == {"image", "label", "index", "mask"}
