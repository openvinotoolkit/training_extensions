# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import copy
import os
import pathlib

import numpy as np
import pytest
from openvino.model_api.models import Model

from otx.algorithms.detection.adapters.openvino.task import (
    OpenVINODetectionInferencer,
    OpenVINODetectionTask,
    OpenVINOMaskInferencer,
    OpenVINORotatedRectInferencer,
)
from otx.algorithms.detection.configs.base import DetectionConfig
from otx.algorithms.detection.utils import generate_label_schema
from otx.api.configuration.helper import create
from otx.api.entities.annotation import AnnotationSceneEntity, AnnotationSceneKind
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.inference_parameters import InferenceParameters
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
from otx.api.usecases.exportable_code.prediction_to_annotation_converter import DetectionBoxToAnnotationConverter
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
        mocker.patch("otx.algorithms.detection.adapters.openvino.task.OpenvinoAdapter")
        mocked_model = mocker.patch.object(Model, "create_model")
        adapter_mock = mocker.Mock(set_callback=mocker.Mock(return_value=None))
        mocked_model.return_value = mocker.MagicMock(spec=Model, inference_adapter=adapter_mock)
        self.ov_inferencer = OpenVINODetectionInferencer(params, label_schema, "")
        self.fake_input = np.full((5, 1), 0.1)

    @e2e_pytest_unit
    def test_pre_process(self):
        """Test pre_process method in OpenVINODetectionInferencer."""
        self.ov_inferencer.model.preprocess.return_value = {"foo": "bar"}
        returned_value = self.ov_inferencer.pre_process(self.fake_input)
        assert returned_value == {"foo": "bar"}

    @e2e_pytest_unit
    def test_predict(self, mocker):
        """Test predict method in OpenVINODetectionInferencer."""
        fake_output = AnnotationSceneEntity(kind=AnnotationSceneKind.ANNOTATION, annotations=[])
        mock_pre_process = mocker.patch.object(OpenVINODetectionInferencer, "pre_process", return_value=("", ""))
        mock_forward = mocker.patch.object(OpenVINODetectionInferencer, "forward")
        mock_converter = mocker.patch.object(
            self.ov_inferencer.converter, "convert_to_annotation", return_value=fake_output
        )
        returned_value, _ = self.ov_inferencer.predict(self.fake_input)

        mock_pre_process.assert_called_once()
        mock_forward.assert_called_once()
        mock_converter.assert_called_once()
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
        mocker.patch("otx.algorithms.detection.adapters.openvino.task.OpenvinoAdapter")
        mocked_model = mocker.patch.object(Model, "create_model")
        adapter_mock = mocker.Mock(set_callback=mocker.Mock(return_value=None))
        mocked_model.return_value = mocker.MagicMock(spec=Model, inference_adapter=adapter_mock)
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
        mocker.patch("otx.algorithms.detection.adapters.openvino.task.OpenvinoAdapter")
        mocked_model = mocker.patch.object(Model, "create_model")
        adapter_mock = mocker.Mock(set_callback=mocker.Mock(return_value=None))
        mocked_model.return_value = mocker.MagicMock(spec=Model, inference_adapter=adapter_mock)
        self.ov_inferencer = OpenVINORotatedRectInferencer(params, label_schema, "")
        self.fake_input = np.full((5, 1), 0.1)

    @e2e_pytest_unit
    def test_pre_process(self):
        """Test pre_process method in RotatedRectInferencer."""
        self.ov_inferencer.model.preprocess.return_value = None, {"foo": "bar"}
        returned_value = self.ov_inferencer.pre_process(self.fake_input)
        assert returned_value == (None, {"foo": "bar"})


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
        mocker.patch("otx.algorithms.detection.adapters.openvino.task.OpenvinoAdapter")
        mocked_model = mocker.patch.object(Model, "create_model")
        adapter_mock = mocker.Mock(set_callback=mocker.Mock(return_value=None))
        mocked_model.return_value = mocker.MagicMock(spec=Model, inference_adapter=adapter_mock)
        ov_inferencer = OpenVINODetectionInferencer(params, label_schema, "")
        ov_inferencer.model.__model__ = "OTX_SSD"
        task_env.model = otx_model
        mocker.patch.object(OpenVINODetectionTask, "load_inferencer", return_value=ov_inferencer)

        self.ov_task = OpenVINODetectionTask(task_env)

    @e2e_pytest_unit
    def test_infer(self, mocker):
        """Test infer method in OpenVINODetectionTask."""
        self.dataset, labels = generate_det_dataset(task_type=TaskType.DETECTION)
        fake_ann_scene = self.dataset[0].annotation_scene
        mock_predict = mocker.patch.object(
            OpenVINODetectionInferencer, "predict", return_value=(fake_ann_scene, (None, None))
        )
        updated_dataset = self.ov_task.infer(self.dataset, InferenceParameters(enable_async_inference=False))

        mock_predict.assert_called()
        for updated in updated_dataset:
            assert updated.annotation_scene.contains_any([LabelEntity(name=labels[0].name, domain="DETECTION")])

    @e2e_pytest_unit
    def test_infer_async(self, mocker):
        """Test async infer method in OpenVINODetectionTask."""
        self.dataset, labels = generate_det_dataset(task_type=TaskType.DETECTION)
        mock_pre_process = mocker.patch.object(
            OpenVINODetectionInferencer, "pre_process", return_value=(None, {"foo", "bar"})
        )
        updated_dataset = self.ov_task.infer(self.dataset, InferenceParameters(enable_async_inference=True))

        mock_pre_process.assert_called()
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
        self.ov_task.deploy(output_model)

        assert output_model.exportable_code is not None

    @e2e_pytest_unit
    def test_optimize(self, mocker, otx_model):
        """Test optimize method in OpenVINODetectionTask."""

        def patch_save_model(model, output_xml):
            with open(output_xml, "wb") as f:
                f.write(b"foo")
            bin_path = pathlib.Path(output_xml).parent / pathlib.Path(str(pathlib.Path(output_xml).stem) + ".bin")
            with open(bin_path, "wb") as f:
                f.write(b"bar")

        dataset, _ = generate_det_dataset(task_type=TaskType.DETECTION)
        output_model = copy.deepcopy(otx_model)
        self.ov_task.model.set_data("openvino.bin", b"foo")
        self.ov_task.model.set_data("openvino.xml", b"bar")
        mocker.patch("otx.algorithms.detection.adapters.openvino.task.ov.Core.read_model", autospec=True)
        mocker.patch("otx.algorithms.detection.adapters.openvino.task.ov.save_model", new=patch_save_model)
        fake_quantize = mocker.patch("otx.algorithms.detection.adapters.openvino.task.nncf.quantize", autospec=True)
        self.ov_task.optimize(OptimizationType.POT, dataset=dataset, output_model=output_model)

        fake_quantize.assert_called_once()
        assert self.ov_task.model.get_data("openvino.bin")
        assert self.ov_task.model.get_data("openvino.xml")
