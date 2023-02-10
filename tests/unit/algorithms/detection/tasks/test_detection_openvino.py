# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import copy
import os

import numpy as np
import pytest
from openvino.model_zoo.model_api.models import Model

import otx.algorithms.detection.tasks.openvino
from otx.algorithms.detection.configs.base import DetectionConfig
from otx.algorithms.detection.tasks.openvino import (
    OpenVINODetectionInferencer,
    OpenVINODetectionTask,
    OpenVINOMaskInferencer,
    OpenVINORotatedRectInferencer,
)
from otx.algorithms.detection.utils import generate_label_schema
from otx.api.configuration.helper import create
from otx.api.entities.annotation import AnnotationSceneEntity, AnnotationSceneKind
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.label import LabelEntity
from otx.api.entities.metrics import Performance, ScoreMetric
from otx.api.entities.model_template import (
    TaskType,
    parse_model_template,
    task_type_to_label_domain,
)
from otx.api.entities.resultset import ResultSetEntity
from otx.api.usecases.evaluation.metrics_helper import MetricsHelper
from otx.api.usecases.tasks.interfaces.optimization_interface import OptimizationType
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.detection.test_helpers import (
    DEFAULT_DET_TEMPLATE_DIR,
    DEFAULT_ISEG_TEMPLATE_DIR,
    generate_det_dataset,
    init_environment,
)


class TestOpenVINODetectionInferencer:
    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        classes = ("rectangle", "ellipse", "triangle")
        self.ov_inferencer = dict()
        task_type = TaskType.DETECTION
        model_template = parse_model_template(os.path.join(DEFAULT_DET_TEMPLATE_DIR, "template.yaml"))
        hyper_parameters = create(model_template.hyper_parameters.data)
        params = DetectionConfig(header=hyper_parameters.header)
        label_schema = generate_label_schema(classes, task_type_to_label_domain(task_type))
        mocker.patch("otx.algorithms.detection.tasks.openvino.OpenvinoAdapter")
        mocked_model = mocker.patch.object(Model, "create_model")
        mocked_model.return_value = mocker.MagicMock(spec=Model)
        self.ov_inferencer = OpenVINODetectionInferencer(params, label_schema, "")
        self.fake_input = np.full((5, 1), 0.1)

    @e2e_pytest_unit
    def test_pre_process(self):
        """Test pre_process method in OpenVINODetectionInferencer."""
        self.ov_inferencer.model.preprocess.return_value = {"foo": "bar"}
        returned_value = self.ov_inferencer.pre_process(self.fake_input)
        assert returned_value == {"foo": "bar"}

    @e2e_pytest_unit
    def test_post_process(self):
        """Test post_process method in OpenVINODetectionInferencer."""
        fake_prediction = {
            "boxes": np.random.rand(1, 5, 5),
            "labels": np.array([[0, 0, 0, 1, 0]]),
            "feature_vector": np.random.rand(1, 320, 1, 1),
            "saliency_map": np.random.rand(1, 2, 6, 8),
        }
        fake_metadata = {"original_shape": (480, 640, 3), "resized_shape": (736, 992, 3)}
        returned_value = self.ov_inferencer.post_process(fake_prediction, fake_metadata)
        assert isinstance(returned_value, AnnotationSceneEntity)

    @e2e_pytest_unit
    def test_predict(self, mocker):
        """Test predict method in OpenVINODetectionInferencer."""
        fake_output = AnnotationSceneEntity(kind=AnnotationSceneKind.ANNOTATION, annotations=[])
        mock_pre_process = mocker.patch.object(OpenVINODetectionInferencer, "pre_process", return_value=("", ""))
        mock_forward = mocker.patch.object(OpenVINODetectionInferencer, "forward")
        mock_post_process = mocker.patch.object(OpenVINODetectionInferencer, "post_process", return_value=fake_output)
        returned_value, _ = self.ov_inferencer.predict(self.fake_input)

        mock_pre_process.assert_called_once()
        mock_forward.assert_called_once()
        mock_post_process.assert_called_once()
        assert returned_value == fake_output

    @e2e_pytest_unit
    def test_forward(self):
        """Test forward method in OpenVINODetectionInferencer."""
        fake_output = {"pred": np.full((5, 1), 0.9)}
        self.ov_inferencer.model.infer_sync.return_value = fake_output
        returned_value = self.ov_inferencer.forward({"image": self.fake_input})
        assert returned_value == fake_output


class TestOpenVINOMaskInferencer:
    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        classes = ("rectangle", "ellipse", "triangle")
        self.ov_inferencer = dict()
        task_type = TaskType.INSTANCE_SEGMENTATION
        model_template = parse_model_template(os.path.join(DEFAULT_ISEG_TEMPLATE_DIR, "template.yaml"))
        hyper_parameters = create(model_template.hyper_parameters.data)
        params = DetectionConfig(header=hyper_parameters.header)
        label_schema = generate_label_schema(classes, task_type_to_label_domain(task_type))
        mocker.patch("otx.algorithms.detection.tasks.openvino.OpenvinoAdapter")
        mocked_model = mocker.patch.object(Model, "create_model")
        mocked_model.return_value = mocker.MagicMock(spec=Model)
        self.ov_inferencer = OpenVINOMaskInferencer(params, label_schema, "")
        self.fake_input = np.full((5, 1), 0.1)

    @e2e_pytest_unit
    def test_pre_process(self):
        """Test pre_process method in OpenVINOMaskInferencer."""
        self.ov_inferencer.model.preprocess.return_value = {"foo": "bar"}
        returned_value = self.ov_inferencer.pre_process(self.fake_input)
        assert returned_value == {"foo": "bar"}


class TestOpenVINORotatedRectInferencer:
    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        classes = ("rectangle", "ellipse", "triangle")
        self.ov_inferencer = dict()
        task_type = TaskType.DETECTION
        model_template = parse_model_template(os.path.join(DEFAULT_DET_TEMPLATE_DIR, "template.yaml"))
        hyper_parameters = create(model_template.hyper_parameters.data)
        params = DetectionConfig(header=hyper_parameters.header)
        label_schema = generate_label_schema(classes, task_type_to_label_domain(task_type))
        mocker.patch("otx.algorithms.detection.tasks.openvino.OpenvinoAdapter")
        mocked_model = mocker.patch.object(Model, "create_model")
        mocked_model.return_value = mocker.MagicMock(spec=Model)
        self.ov_inferencer = OpenVINORotatedRectInferencer(params, label_schema, "")
        self.fake_input = np.full((5, 1), 0.1)

    @e2e_pytest_unit
    def test_pre_process(self):
        """Test pre_process method in RotatedRectInferencer."""
        self.ov_inferencer.model.preprocess.return_value = {"foo": "bar"}
        returned_value = self.ov_inferencer.pre_process(self.fake_input)
        assert returned_value == {"foo": "bar"}


class TestOpenVINODetectionTask:
    @pytest.fixture(autouse=True)
    def setup(self, mocker, otx_model) -> None:

        classes = ("rectangle", "ellipse", "triangle")
        self.ov_inferencer = dict()
        task_type = TaskType.DETECTION

        model_template = parse_model_template(os.path.join(DEFAULT_DET_TEMPLATE_DIR, "template.yaml"))
        hyper_parameters = create(model_template.hyper_parameters.data)
        label_schema = generate_label_schema(classes, task_type_to_label_domain(task_type))
        task_env = init_environment(hyper_parameters, model_template, task_type=task_type)
        params = DetectionConfig(header=hyper_parameters.header)
        mocker.patch("otx.algorithms.detection.tasks.openvino.OpenvinoAdapter")
        mocked_model = mocker.patch.object(Model, "create_model")
        mocked_model.return_value = mocker.MagicMock(spec=Model)
        ov_inferencer = OpenVINODetectionInferencer(params, label_schema, "")
        ov_inferencer.model.__model__ = "OTX_SSD"
        task_env.model = otx_model
        mocker.patch.object(OpenVINODetectionTask, "load_inferencer", return_value=ov_inferencer)
        mocker.patch.object(OpenVINODetectionTask, "load_config", return_value={})

        self.ov_task = OpenVINODetectionTask(task_env)

    @e2e_pytest_unit
    def test_infer(self, mocker):
        """Test infer method in OpenVINODetectionTask."""
        self.dataset, labels = generate_det_dataset(task_type=TaskType.DETECTION)
        fake_ann_scene = self.dataset[0].annotation_scene
        mock_predict = mocker.patch.object(
            OpenVINODetectionInferencer, "predict", return_value=(fake_ann_scene, (None, None))
        )
        updated_dataset = self.ov_task.infer(self.dataset)

        mock_predict.assert_called()
        for updated in updated_dataset:
            assert updated.annotation_scene.contains_any([LabelEntity(name=labels[0].name, domain="DETECTION")])

    @e2e_pytest_unit
    def test_evaluate(self, mocker):
        """Test evaluate method in OpenVINODetectionTask."""
        result_set = ResultSetEntity(
            model=None,
            ground_truth_dataset=DatasetEntity(),
            prediction_dataset=DatasetEntity(),
        )
        fake_metrics = mocker.patch("otx.api.usecases.evaluation.f_measure.FMeasure", autospec=True)
        fake_metrics.get_performance.return_value = Performance(
            score=ScoreMetric(name="fake", value=0.1), dashboard_metrics="mAP"
        )
        mocker.patch.object(MetricsHelper, "compute_f_measure", return_value=fake_metrics)
        self.ov_task.evaluate(result_set)

        assert result_set.performance.score.value == 0.1

    @e2e_pytest_unit
    def test_deploy(self, otx_model):
        """Test deploy method in OpenVINODetectionTask."""
        output_model = copy.deepcopy(otx_model)
        self.ov_task.model.set_data("openvino.bin", b"foo")
        self.ov_task.model.set_data("openvino.xml", b"bar")
        self.ov_task.config = {"tiling_parameters": None}
        self.ov_task.deploy(output_model)

        assert output_model.exportable_code is not None

    @e2e_pytest_unit
    def test_optimize(self, mocker, otx_model):
        """Test optimize method in OpenVINODetectionTask."""

        def patch_save_model(model, dir_path, model_name):
            with open(f"{dir_path}/{model_name}.xml", "wb") as f:
                f.write(b"foo")
            with open(f"{dir_path}/{model_name}.bin", "wb") as f:
                f.write(b"bar")

        dataset, _ = generate_det_dataset(task_type=TaskType.DETECTION)
        output_model = copy.deepcopy(otx_model)
        self.ov_task.model.set_data("openvino.bin", b"foo")
        self.ov_task.model.set_data("openvino.xml", b"bar")
        mocker.patch("otx.algorithms.detection.tasks.openvino.load_model", autospec=True)
        mocker.patch("otx.algorithms.detection.tasks.openvino.create_pipeline", autospec=True)
        mocker.patch("otx.algorithms.detection.tasks.openvino.save_model", new=patch_save_model)
        spy_compress = mocker.spy(otx.algorithms.detection.tasks.openvino, "compress_model_weights")
        self.ov_task.optimize(OptimizationType.POT, dataset=dataset, output_model=output_model)

        spy_compress.assert_called_once()
        assert self.ov_task.model.get_data("openvino.bin")
        assert self.ov_task.model.get_data("openvino.xml")
