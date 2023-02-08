"""Tests the inference callback on a dummy lightning module."""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import pytorch_lightning as pl

from otx.algorithms.anomaly.adapters.anomalib.callbacks import AnomalyInferenceCallback
from otx.api.entities.model_template import TaskType
from tests.unit.algorithms.anomaly.helpers.dummy_dataset import MockDataModule
from tests.unit.algorithms.anomaly.helpers.dummy_model import DummyModel


class TestInferenceCallback:
    @pytest.mark.parametrize(
        "task_type", [TaskType.ANOMALY_CLASSIFICATION, TaskType.ANOMALY_DETECTION, TaskType.ANOMALY_SEGMENTATION]
    )
    def test_inference_callback(self, task_type):
        datamodule = MockDataModule(task_type)
        model = DummyModel()
        labels = datamodule.labels
        callback = AnomalyInferenceCallback(datamodule.dataset.dataset, labels, task_type)
        trainer = pl.Trainer(logger=False, callbacks=[callback])
        result = trainer.predict(model, datamodule=datamodule)
        assert result[0]["pred_labels"].item() == 1.0
