"""Test OpenVINO model."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import pytest
from anomalib import TaskType as AnomalibTaskType
from openvino.model_api.models.anomaly import AnomalyResult

from otx.algo.anomaly.openvino_model import AnomalyOpenVINO
from otx.core.types.task import OTXTaskType


class TestOpenVINOModel:
    """Test OpenVINO model."""

    @pytest.fixture(scope="class")
    def model_path(self) -> str:
        """Get absolute path to the model from assets directory."""
        return str(Path(__file__).parent / "assets" / "dummy.xml")

    @pytest.mark.parametrize(
        "task",
        [OTXTaskType.ANOMALY_CLASSIFICATION, OTXTaskType.ANOMALY_DETECTION, OTXTaskType.ANOMALY_SEGMENTATION],
    )
    def test_create_model(self, task: OTXTaskType, model_path: str):
        """Test create model."""
        model = AnomalyOpenVINO(model_name=model_path)

        # check conversion of task type
        model.task = task
        match task:
            case OTXTaskType.ANOMALY_CLASSIFICATION:
                assert model.task == AnomalibTaskType.CLASSIFICATION
            case OTXTaskType.ANOMALY_DETECTION:
                assert model.task == AnomalibTaskType.DETECTION
            case OTXTaskType.ANOMALY_SEGMENTATION:
                assert model.task == AnomalibTaskType.SEGMENTATION
            case _:
                pytest.fail("Invalid task type")

        assert model.model_type == "AnomalyDetection"
        assert model.image_threshold == 0.7
        assert model.pixel_threshold == 0.1

    @pytest.mark.parametrize(
        "task",
        [OTXTaskType.ANOMALY_CLASSIFICATION, OTXTaskType.ANOMALY_DETECTION, OTXTaskType.ANOMALY_SEGMENTATION],
    )
    def test_update_metrics(self, task: OTXTaskType, model_path: str):
        """Test update metrics."""
        # test when pred_mask and anomaly_map are None
        # should compute only image_metrics
        model = AnomalyOpenVINO(model_name=model_path)
        model.task = task
        model.configure_callbacks()
        dummy_results = [
            AnomalyResult(anomaly_map=None, pred_mask=None, pred_score=0.7, pred_label="Anomaly"),
            AnomalyResult(anomaly_map=None, pred_mask=None, pred_score=0.0, pred_label="Normal"),
        ]
        model._update_metrics(dummy_results)
        image_metrics = model.image_metrics.compute()
        assert len(image_metrics) == 2
        for key, value in image_metrics.items():
            assert key in ["image_AUROC", "image_F1Score"]
            assert value == 1.0
        assert model.pixel_metrics._update_called is False

        # test when pred_mask and anomaly_map are not None
        # should compute both image_metrics and pixel_metrics
        dummy_results = [
            AnomalyResult(
                anomaly_map=np.array([255.0]),
                pred_mask=np.ones(1, dtype=bool),
                pred_score=0.7,
                pred_label="Anomaly",
            ),
            AnomalyResult(
                anomaly_map=np.zeros(1),
                pred_mask=np.zeros(1, dtype=bool),
                pred_score=0.0,
                pred_label="Normal",
            ),
        ]
        model._update_metrics(dummy_results)
        assert len(image_metrics) == 2
        for key, value in image_metrics.items():
            assert key in ["image_AUROC", "image_F1Score"]
            assert value == 1.0
        pixel_metrics = model.pixel_metrics.compute()
        if task != OTXTaskType.ANOMALY_CLASSIFICATION:
            assert len(pixel_metrics) == 2
            for key, value in pixel_metrics.items():
                assert key in ["pixel_AUROC", "pixel_F1Score"]
                assert value == 1.0
