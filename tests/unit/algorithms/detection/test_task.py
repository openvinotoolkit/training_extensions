"""Test otx detection task."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os

import numpy as np
import pytest
from mmcv import Config

from otx.algorithms.detection.task import OTXDetectionTask
from otx.algorithms.detection.utils import generate_label_schema
from otx.api.configuration.helper import create
from otx.api.entities.model_template import (
    TaskType,
    parse_model_template,
    task_type_to_label_domain,
)
from tests.unit.algorithms.detection.test_helpers import (
    DEFAULT_DET_TEMPLATE_DIR,
    generate_det_dataset,
    init_environment,
)


class MockOTXDetectionTask(OTXDetectionTask):
    def _infer_model(*args, **kwargs):
        return zip([[np.array([[0, 0, 1, 1, 1]])]] * 50, np.ndarray([50, 472, 1, 1]), [None] * 50), 1.0

    def _train_model(*args, **kwargs):
        return {"final_ckpt": "dummy.pth"}

    def explain(*args, **kwargs):
        pass

    def export(*args, **kwargs):
        pass


class MockOTXIsegTask(OTXDetectionTask):
    def _infer_model(*args, **kwargs):
        predictions = zip([[np.array([[0, 0, 1, 1, 1]])]] * 50, [np.array([[0, 0, 1, 1, 1, 1, 1]])] * 50)
        return (
            zip(
                predictions,
                np.ndarray([50, 472, 1, 1]),
                [None] * 50,
            ),
            1.0,
        )

    def _train_model(*args, **kwargs):
        return {"final_ckpt": "dummy.pth"}

    def explain(*args, **kwargs):
        pass

    def export(*args, **kwargs):
        pass


class MockModel:
    class _Configuration:
        def __init__(self, label_schema):
            self.label_schema = label_schema

        def get_label_schema(self):
            return self.label_schema

    def __init__(self):
        self.model_adapters = ["weights.pth"]
        self.data = np.ndarray(1)

        classes = ("rectangle", "ellipse", "triangle")
        label_schema = generate_label_schema(classes, task_type_to_label_domain(TaskType.DETECTION))

        self.configuration = self._Configuration(label_schema)

    def get_data(self, name):
        return self.data

    def set_data(self, *args, **kwargs):
        return


class TestOTXDetectionTask:
    @pytest.fixture(autouse=True)
    def setup(self, mocker):
        model_template = parse_model_template(os.path.join(DEFAULT_DET_TEMPLATE_DIR, "template.yaml"))
        hyper_parameters = create(model_template.hyper_parameters.data)
        task_env = init_environment(hyper_parameters, model_template, task_type=TaskType.DETECTION)

        self.det_task = MockOTXDetectionTask(task_env)

    def test_load_model_ckpt(self, mocker):
        mocker.patch(
            "torch.load",
            return_value={
                "anchors": [1],
                "confidence_threshold": 0.1,
                "config": {
                    "tiling_parameters": {
                        "enable_tiling": {"value": True},
                        "tile_size": {"value": 256},
                        "tile_overlap": {"value": 0},
                        "tile_max_number": {"value": 500},
                    }
                },
            },
        )

        self.det_task._load_model_ckpt(MockModel())

        assert self.det_task._anchors == [1]
        assert self.det_task._hyperparams.tiling_parameters.enable_tiling is True
        assert self.det_task._hyperparams.tiling_parameters.tile_size == 256
        assert self.det_task._hyperparams.tiling_parameters.tile_overlap == 0
        assert self.det_task._hyperparams.tiling_parameters.tile_max_number == 500

    def test_train(self, mocker):
        dataset = generate_det_dataset(TaskType.DETECTION, 50)[0]
        mocker.patch("torch.load", return_value=np.ndarray([1]))
        self.det_task.train(dataset, MockModel())
        assert self.det_task._model_ckpt == "dummy.pth"

    def test_infer(self):
        dataset = generate_det_dataset(TaskType.DETECTION, 50)[0]
        predicted_dataset = self.det_task.infer(dataset.with_empty_annotations())
        assert predicted_dataset[0].annotation_scene.annotations[0].shape.x1 == 0.0
        assert predicted_dataset[0].annotation_scene.annotations[0].shape.y1 == 0.0

    def test_evaluate(self, mocker):
        class _MockMetric:
            f_measure = Config({"value": 1.0})

            def get_performance(self):
                return 1.0

        class _MockResultEntity:
            performance = 0.0

        mocker.patch(
            "otx.algorithms.detection.task.MetricsHelper.compute_f_measure",
            return_value=_MockMetric(),
        )

        _result_entity = _MockResultEntity()
        self.det_task.evaluate(_result_entity)
        assert _result_entity.performance == 1.0
