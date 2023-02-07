# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os

import numpy as np
import pytest
from openvino.model_zoo.model_api.models import Model

from otx.algorithms.detection.configs.base import DetectionConfig
from otx.algorithms.detection.tasks.openvino import OpenVINODetectionInferencer
from otx.algorithms.detection.utils import generate_label_schema
from otx.api.configuration.helper import create
from otx.api.entities.annotation import AnnotationSceneEntity, AnnotationSceneKind
from otx.api.entities.model_template import (
    TaskType,
    parse_model_template,
    task_type_to_label_domain,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.detection.test_helpers import (
    DEFAULT_DET_TEMPLATE_DIR,
    DEFAULT_ISEG_TEMPLATE_DIR,
)


class TestOpenVINODetectionInferencer:
    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        classes = ("rectangle", "ellipse", "triangle")
        self.ov_inferencer = dict()
        for template, task_type in zip(
            [DEFAULT_DET_TEMPLATE_DIR, DEFAULT_ISEG_TEMPLATE_DIR], (TaskType.DETECTION, TaskType.INSTANCE_SEGMENTATION)
        ):
            model_template = parse_model_template(os.path.join(template, "template.yaml"))
            hyper_parameters = create(model_template.hyper_parameters.data)
            params = DetectionConfig(header=hyper_parameters.header)
            label_schema = generate_label_schema(classes, task_type_to_label_domain(task_type))
            mocker.patch("otx.algorithms.detection.tasks.openvino.OpenvinoAdapter")
            mocker.patch.object(Model, "create_model")
            self.ov_inferencer[task_type] = OpenVINODetectionInferencer(params, label_schema, "")
        self.fake_input = np.full((5, 1), 0.1)

    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "task_type",
        [TaskType.DETECTION, TaskType.INSTANCE_SEGMENTATION],
    )
    def test_pre_process(self, task_type):
        self.ov_inferencer[task_type].model.preprocess.return_value = {"foo": "bar"}
        returned_value = self.ov_inferencer[task_type].pre_process(self.fake_input)
        assert returned_value == {"foo": "bar"}

    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "task_type",
        [TaskType.DETECTION, TaskType.INSTANCE_SEGMENTATION],
    )
    def test_post_process(self, task_type):
        fake_prediction = {
            "boxes": np.random.rand(1, 5, 5),
            "labels": np.array([[0, 0, 0, 1, 0]]),
            "feature_vector": np.random.rand(1, 320, 1, 1),
            "saliency_map": np.random.rand(1, 2, 6, 8),
        }
        fake_metadata = {"original_shape": (480, 640, 3), "resized_shape": (736, 992, 3)}
        returned_value = self.ov_inferencer[task_type].post_process(fake_prediction, fake_metadata)
        assert isinstance(returned_value, AnnotationSceneEntity)

    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "task_type",
        [TaskType.DETECTION, TaskType.INSTANCE_SEGMENTATION],
    )
    def test_predict(self, task_type, mocker):
        fake_output = AnnotationSceneEntity(kind=AnnotationSceneKind.ANNOTATION, annotations=[])
        mock_pre_process = mocker.patch.object(OpenVINODetectionInferencer, "pre_process", return_value=("", ""))
        mock_forward = mocker.patch.object(OpenVINODetectionInferencer, "forward")
        mock_post_process = mocker.patch.object(OpenVINODetectionInferencer, "post_process", return_value=fake_output)
        returned_value, _ = self.ov_inferencer[task_type].predict(self.fake_input)

        mock_pre_process.assert_called_once()
        mock_forward.assert_called_once()
        mock_post_process.assert_called_once()
        assert returned_value == fake_output

    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "task_type",
        [TaskType.DETECTION, TaskType.INSTANCE_SEGMENTATION],
    )
    def test_forward(self, task_type):
        fake_output = {"pred": np.full((5, 1), 0.9)}
        self.ov_inferencer[task_type].model.infer_sync.return_value = fake_output
        returned_value = self.ov_inferencer[task_type].forward({"image": self.fake_input})
        assert returned_value == fake_output
