# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os

import numpy as np
import pytest

from otx.algorithms.common.adapters.mmcv.hooks import OTXLoggerHook
from otx.algorithms.common.tasks import BaseTask
from otx.algorithms.detection.tasks import DetectionTrainTask
from otx.api.configuration.helper import create
from otx.api.entities.metrics import NullPerformance, Performance, ScoreMetric
from otx.api.entities.model_template import TaskType, parse_model_template
from otx.api.usecases.evaluation.metrics_helper import MetricsHelper
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.detection.test_helpers import (
    DEFAULT_DET_TEMPLATE_DIR,
    generate_det_dataset,
    init_environment,
)


class TestOTXDetTaskTrain:
    @pytest.fixture(autouse=True)
    def setup(self, otx_model, tmp_dir_path) -> None:
        model_template = parse_model_template(os.path.join(DEFAULT_DET_TEMPLATE_DIR, "template.yaml"))
        hyper_parameters = create(model_template.hyper_parameters.data)
        task_env = init_environment(hyper_parameters, model_template)
        self.model = otx_model
        self.train_task = DetectionTrainTask(task_env, output_path=str(tmp_dir_path))

    @e2e_pytest_unit
    def test_save_model(self, mocker):
        """Test save_model method in OTXDetTaskTrain."""
        mocker.patch("torch.load", return_value="")
        self.train_task.save_model(self.model)

        assert self.model.get_data("weights.pth")
        assert self.model.get_data("label_schema.json")

    @e2e_pytest_unit
    def test_train(self, mocker):
        """Test train method in OTXDetTaskTrain."""
        self.dataset, _ = generate_det_dataset(task_type=TaskType.DETECTION)
        mock_lcurve_val = OTXLoggerHook.Curve()
        mock_lcurve_val.x = [0, 1]
        mock_lcurve_val.y = [0.1, 0.2]
        mock_run_task = mocker.patch.object(BaseTask, "_run_task", return_value={"final_ckpt": ""})
        self.train_task._learning_curves = {"val/mAP": mock_lcurve_val}
        mocker.patch.object(DetectionTrainTask, "save_model")

        fake_prediction = [[np.array([[0, 0, 32, 24, 0.55]], dtype=np.float32)]]
        fake_feature_vectors = [np.zeros((1, 1, 1))]
        fake_saliency_maps = [None]
        mocker.patch.object(
            DetectionTrainTask,
            "_infer_detector",
            return_value=(zip(fake_prediction, fake_feature_vectors, fake_saliency_maps), 1.0),
        )
        fake_metrics = mocker.patch("otx.api.usecases.evaluation.f_measure.FMeasure", autospec=True)
        fake_metrics.get_performance.return_value = Performance(
            score=ScoreMetric(name="fake", value=0.1), dashboard_metrics=["mAP"]
        )
        mocker.patch.object(MetricsHelper, "compute_f_measure", return_value=fake_metrics)

        self.train_task.train(self.dataset, self.model)

        mock_run_task.assert_called_once()
        assert self.model.performance != NullPerformance()
        assert self.model.performance.score.value == 0.1

    @e2e_pytest_unit
    def test_cancel_training(self):
        """Test cancel_training method in OTXDetTaskTrain."""
        self.train_task.cancel_training()
        assert self.train_task._should_stop is True
