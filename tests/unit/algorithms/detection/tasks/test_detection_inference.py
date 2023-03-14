# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os

import numpy as np
import pytest

from otx.algorithms.common.tasks import BaseTask
from otx.algorithms.detection.tasks import DetectionInferenceTask
from otx.api.configuration.helper import create
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.label import Domain, LabelEntity
from otx.api.entities.metrics import Performance, ScoreMetric
from otx.api.entities.model import ModelPrecision
from otx.api.entities.model_template import TaskType, parse_model_template
from otx.api.entities.resultset import ResultSetEntity
from otx.api.usecases.evaluation.metrics_helper import MetricsHelper
from otx.api.usecases.tasks.interfaces.export_interface import ExportType
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.detection.test_helpers import (
    DEFAULT_DET_TEMPLATE_DIR,
    DEFAULT_ISEG_TEMPLATE_DIR,
    generate_det_dataset,
    init_environment,
)


class TestOTXDetectionTaskInference:
    @pytest.fixture(autouse=True)
    def setup(self, otx_model, tmp_dir_path) -> None:
        self.inference_task = dict()
        for template, task_type in zip(
            [DEFAULT_DET_TEMPLATE_DIR, DEFAULT_ISEG_TEMPLATE_DIR], (TaskType.DETECTION, TaskType.INSTANCE_SEGMENTATION)
        ):
            model_template = parse_model_template(os.path.join(template, "template.yaml"))
            hyper_parameters = create(model_template.hyper_parameters.data)
            task_env = init_environment(hyper_parameters, model_template, task_type=task_type)
            inference_task = DetectionInferenceTask(task_env, output_path=str(tmp_dir_path))
            self.inference_task[task_type] = inference_task
        self.output_path = str(tmp_dir_path)
        self.model = otx_model

    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "task_type, domain",
        [(TaskType.DETECTION, Domain.DETECTION), (TaskType.INSTANCE_SEGMENTATION, Domain.INSTANCE_SEGMENTATION)],
    )
    def test_infer(self, task_type, domain, mocker):
        """Test infer method in DetectionInferenceTask."""
        dataset, labels = generate_det_dataset(task_type=task_type)
        if task_type == TaskType.DETECTION:
            fake_prediction = [[np.random.rand(1, 5)]]
        elif task_type == TaskType.INSTANCE_SEGMENTATION:
            sample = dataset[0]
            fake_prediction = [
                (
                    [np.random.rand(1, 5)],
                    [
                        [
                            {
                                "size": [sample.width, sample.height],
                                "counts": b"<4601OO11OO100O100O",
                            }
                        ]
                    ],
                )
            ]
        fake_feature_vectors = [np.zeros((1, 1, 1))]
        fake_saliency_maps = [None]
        fake_results = dict(
            outputs=dict(
                metric=1.0,
                detections=fake_prediction,
                feature_vectors=fake_feature_vectors,
                saliency_maps=fake_saliency_maps,
            )
        )
        mock_run_task = mocker.patch.object(BaseTask, "_run_task", return_value=fake_results)
        updated_dataset = self.inference_task[task_type].infer(dataset, None)
        mock_run_task.assert_called_once()
        for updated in updated_dataset:
            assert updated.annotation_scene.contains_any([LabelEntity(name=labels[0].name, domain=domain)])

    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "task_type",
        [TaskType.DETECTION, TaskType.INSTANCE_SEGMENTATION],
    )
    def test_evaluate(self, task_type, mocker):
        """Test evaluate method in DetectionInferenceTask."""
        result_set = ResultSetEntity(
            model=self.model,
            ground_truth_dataset=DatasetEntity(),
            prediction_dataset=DatasetEntity(),
        )
        fake_metrics = mocker.patch("otx.api.usecases.evaluation.f_measure.FMeasure", autospec=True)
        fake_metrics.get_performance.return_value = Performance(
            score=ScoreMetric(name="fake", value=0.1), dashboard_metrics="mAP"
        )
        mocker.patch.object(MetricsHelper, "compute_f_measure", return_value=fake_metrics)
        self.inference_task[task_type].evaluate(result_set)
        assert result_set.performance.score.value == 0.1

    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "task_type",
        [TaskType.DETECTION, TaskType.INSTANCE_SEGMENTATION],
    )
    @pytest.mark.parametrize("precision", [ModelPrecision.FP16, ModelPrecision.FP32])
    def test_export(self, task_type, mocker, precision: ModelPrecision):
        """Test export method in DetectionInferenceTask, expected RuntimeError without model file."""
        fake_output = {"outputs": {"bin": None, "xml": None}}
        mocker.patch("otx.algorithms.detection.tasks.inference.embed_ir_model_data", return_value=None)
        mock_run_task = mocker.patch.object(BaseTask, "_run_task", return_value=fake_output)

        with pytest.raises(RuntimeError):
            self.inference_task[task_type].export(ExportType.OPENVINO, self.model, precision)
            mock_run_task.assert_called_once()

    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "task_type",
        [TaskType.DETECTION, TaskType.INSTANCE_SEGMENTATION],
    )
    @pytest.mark.parametrize("precision", [ModelPrecision.FP16, ModelPrecision.FP32])
    def test_export_with_model_files(self, task_type, mocker, precision: ModelPrecision):
        """Test export method in DetectionInferenceTask."""
        with open(f"{self.output_path}/model.xml", "wb") as f:
            f.write(b"foo")
        with open(f"{self.output_path}/model.bin", "wb") as f:
            f.write(b"bar")

        fake_output = {"outputs": {"bin": f"{self.output_path}/model.xml", "xml": f"{self.output_path}/model.bin"}}
        mocker.patch("otx.algorithms.detection.tasks.inference.embed_ir_model_data", return_value=None)
        mock_run_task = mocker.patch.object(BaseTask, "_run_task", return_value=fake_output)
        self.inference_task[task_type].export(ExportType.OPENVINO, self.model, precision)

        mock_run_task.assert_called_once()
        assert self.model.get_data("openvino.bin")
        assert self.model.get_data("openvino.xml")

    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "task_type",
        [TaskType.DETECTION, TaskType.INSTANCE_SEGMENTATION],
    )
    def test_unload(self, task_type, mocker):
        """Test unload method in DetectionInferenceTask."""
        mock_cleanup = mocker.patch.object(BaseTask, "cleanup")
        self.inference_task[task_type].unload()
        mock_cleanup.assert_called_once()
