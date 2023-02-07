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


class TestOTXDetTaskInference:
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
        dataset, labels = generate_det_dataset(task_type=task_type)
        if task_type == TaskType.DETECTION:
            fake_prediction = [[np.array([[0, 0, 32, 24, 0.55]], dtype=np.float32)]]
        elif task_type == TaskType.INSTANCE_SEGMENTATION:
            sample = dataset[0]
            # TODO [Jihwan]: Resolve length issue
            fake_prediction = [
                (
                    [np.array([[8, 5, 10, 20, 0.90]], dtype=np.float32)],
                    [
                        [
                            {
                                "size": [sample.width, sample.height],
                                "counts": b"""WVo1c0Z>4E_OSBd0l=81SOSBe0U>O1N3N2N3EfAI\\>3;NU?7
                                            i]O8VFCUN5_;c0VFYO[N4];i0SFTO_N5X;o0VFlNbN5V;R1WFiN
                                            dN4T;V1WFdNfN6Q;Z1XF^NjN5m:b1gFYNV9m1k16J4M3M2N3L6D
                                            <WEoLZ9T3SFQMBOZ:Q3RF^Mm9d2gEhL2e0W:d2cEkL5a0W:n2iE
                                            PMX:P3k02M2O2O1N1O1O2N2M3O2M3N1N2O0O2N1K6N1O100O101
                                            N10000000001O1O2N1O2N2N2N2N1O001O00001O000000000000
                                            000000000000000000000000000000000000000000000000000
                                            0000000000000000000O1000000000000O100O1O1001O00001O
                                            H\\FTKd9k4^FTKb9l4_FSKa9m4900O100000000000000000000
                                            00O100O1O1N200O100O1O2N1O1O2N1eKbET4^:iKeEV4d:O0O2N
                                            1N2O1O2O0O2N1N3M5L3M3M1O1O1O1O1N2N3TMWD^2n;[MUD\\2K
                                            iMZ<U2jCgMY<V2:N;E7H6J3N10000O1000\\OfNZC0MX1i<jNXC
                                            0NV1c=iN\\Bb0Z>_O_i\\4""",
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
    def test_export(self, task_type, mocker):
        fake_output = {"outputs": {"bin": None, "xml": None}}
        mock_run_task = mocker.patch.object(BaseTask, "_run_task", return_value=fake_output)

        with pytest.raises(RuntimeError):
            self.inference_task[task_type].export(ExportType.OPENVINO, self.model)
            mock_run_task.assert_called_once()

    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "task_type",
        [TaskType.DETECTION, TaskType.INSTANCE_SEGMENTATION],
    )
    def test_export_with_model_files(self, task_type, mocker):
        with open(f"{self.output_path}/model.xml", "wb") as f:
            f.write(b"foo")
        with open(f"{self.output_path}/model.bin", "wb") as f:
            f.write(b"bar")

        fake_output = {"outputs": {"bin": f"{self.output_path}/model.xml", "xml": f"{self.output_path}/model.bin"}}
        mock_run_task = mocker.patch.object(BaseTask, "_run_task", return_value=fake_output)
        self.inference_task[task_type].export(ExportType.OPENVINO, self.model)

        mock_run_task.assert_called_once()
        assert self.model.get_data("openvino.bin")
        assert self.model.get_data("openvino.xml")

    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "task_type",
        [TaskType.DETECTION, TaskType.INSTANCE_SEGMENTATION],
    )
    def test_unload(self, task_type, mocker):
        mock_cleanup = mocker.patch.object(BaseTask, "cleanup")
        self.inference_task[task_type].unload()
        mock_cleanup.assert_called_once()
