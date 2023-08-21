# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import copy
import pathlib

import numpy as np
import pytest
from openvino.model_api.models import Model

import otx.algorithms.classification.adapters.openvino.task
from otx.algorithms.classification.adapters.openvino.task import (
    ClassificationOpenVINOInferencer,
    ClassificationOpenVINOTask,
)
from otx.algorithms.classification.configs.base import ClassificationConfig
from otx.api.configuration.configurable_parameters import ConfigurableParameters
from otx.api.entities.annotation import (
    Annotation,
    AnnotationSceneEntity,
    AnnotationSceneKind,
)
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.inference_parameters import InferenceParameters
from otx.api.entities.label_schema import LabelSchemaEntity
from otx.api.entities.metrics import Performance, ScoreMetric
from otx.api.entities.model import ModelConfiguration, ModelEntity
from otx.api.entities.resultset import ResultSetEntity
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.shapes.rectangle import Rectangle
from otx.api.usecases.evaluation.metrics_helper import MetricsHelper
from otx.api.usecases.tasks.interfaces.optimization_interface import OptimizationType
from otx.api.utils.shape_factory import ShapeFactory
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.classification.test_helper import (
    DEFAULT_CLS_TEMPLATE,
    init_environment,
    setup_configurable_parameters,
)


@pytest.fixture
def otx_model():
    model_configuration = ModelConfiguration(
        configurable_parameters=ConfigurableParameters(header="header", description="description"),
        label_schema=LabelSchemaEntity(),
    )
    return ModelEntity(train_dataset=DatasetEntity(), configuration=model_configuration)


class TestOpenVINOClassificationInferencer:
    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        hyper_parameters, model_template = setup_configurable_parameters(DEFAULT_CLS_TEMPLATE)
        cls_params = ClassificationConfig(header=hyper_parameters.header)
        environment, dataset = init_environment(hyper_parameters, model_template)
        self.label_schema = environment.label_schema
        mocker.patch("otx.algorithms.classification.adapters.openvino.task.OpenvinoAdapter")
        mocker.patch.object(Model, "create_model")
        self.cls_ov_inferencer = ClassificationOpenVINOInferencer(cls_params, self.label_schema, "")
        model_path = "otx.algorithms.classification.adapters.openvino.model_wrappers.openvino_models.OTXClassification"
        self.cls_ov_inferencer.model = mocker.patch(model_path, autospec=True)
        self.fake_input = np.random.rand(3, 224, 224)

    @e2e_pytest_unit
    def test_pre_process(self):
        self.cls_ov_inferencer.model.preprocess.return_value = {"foo": "bar"}
        returned_value = self.cls_ov_inferencer.pre_process(self.fake_input)

        assert returned_value == {"foo": "bar"}

    @e2e_pytest_unit
    def test_post_process(self):
        fake_feature_vector = np.random.rand(1, 100)
        fake_logits = np.random.rand(1, 2)
        fake_prediction = {
            "logits": fake_logits,
            "feature_vector": fake_feature_vector,
            "saliency_map": self.fake_input,
        }
        fake_metadata = {"original_shape": (254, 320, 3), "resized_shape": (224, 224, 3)}
        self.cls_ov_inferencer.model.postprocess.return_value = [[0, 0.87], [1, 0.13]]
        returned_value = self.cls_ov_inferencer.post_process(fake_prediction, fake_metadata)

        assert len(returned_value.annotations[0].get_labels()) > 0
        assert len(returned_value.annotations[0].get_labels()) == len(self.label_schema.get_labels(include_empty=False))

    @e2e_pytest_unit
    def test_predict(self, mocker):
        fake_output = AnnotationSceneEntity(kind=AnnotationSceneKind.ANNOTATION, annotations=[])
        mock_pre_process = mocker.patch.object(ClassificationOpenVINOInferencer, "pre_process", return_value=("", ""))
        mock_forward = mocker.patch.object(ClassificationOpenVINOInferencer, "forward")
        self.cls_ov_inferencer.model.postprocess_aux_outputs.return_value = ("", "", "", "")
        mock_post_process = mocker.patch.object(
            ClassificationOpenVINOInferencer, "post_process", return_value=fake_output
        )
        returned_value = self.cls_ov_inferencer.predict(self.fake_input)

        mock_pre_process.assert_called_once()
        mock_forward.assert_called_once()
        mock_post_process.assert_called_once()
        assert returned_value == (fake_output, "", "", "", "")

    @e2e_pytest_unit
    def test_forward(self):
        fake_output = {"logits": np.random.rand(1, 2)}
        self.cls_ov_inferencer.model.infer_sync.return_value = fake_output
        returned_value = self.cls_ov_inferencer.forward({"image": self.fake_input})
        assert returned_value == fake_output


class TestOpenVINOClassificationTask:
    @pytest.fixture(autouse=True)
    def setup(self, mocker, otx_model) -> None:
        hyper_parameters, model_template = setup_configurable_parameters(DEFAULT_CLS_TEMPLATE)
        cls_params = ClassificationConfig(header=hyper_parameters.header)
        self.task_env, self.dataset = init_environment(params=hyper_parameters, model_template=model_template)
        self.label_schema = self.task_env.label_schema
        mocker.patch("otx.algorithms.classification.adapters.openvino.task.OpenvinoAdapter")
        mocker.patch.object(Model, "create_model")
        cls_ov_inferencer = ClassificationOpenVINOInferencer(cls_params, self.label_schema, "")
        self.task_env.model = otx_model
        mocker.patch.object(ClassificationOpenVINOTask, "load_inferencer", return_value=cls_ov_inferencer)
        self.cls_ov_task = ClassificationOpenVINOTask(self.task_env)
        self.labels = self.label_schema.get_labels(include_empty=True)
        fake_annotation = [
            Annotation(
                Rectangle.generate_full_box(),
                id=0,
                labels=[ScoredLabel(label, probability=1.0) for label in self.labels],
            )
        ]
        self.fake_ann_scene = AnnotationSceneEntity(kind=AnnotationSceneKind.ANNOTATION, annotations=fake_annotation)
        self.fake_input = mocker.MagicMock()

    @e2e_pytest_unit
    def test_infer(self, mocker):
        mock_predict = mocker.patch.object(
            ClassificationOpenVINOInferencer,
            "predict",
            return_value=(self.fake_ann_scene, np.array([0, 1]), self.fake_input, self.fake_input, self.fake_input),
        )
        mocker.patch.object(ShapeFactory, "shape_produces_valid_crop", return_value=True)
        updated_dataset = self.cls_ov_task.infer(
            self.dataset, InferenceParameters(enable_async_inference=False, is_evaluation=True)
        )

        mock_predict.assert_called()
        for updated in updated_dataset:
            assert updated.annotation_scene.contains_any(self.labels)

    @e2e_pytest_unit
    def test_infer_async(self, mocker):
        mocker.patch.object(ShapeFactory, "shape_produces_valid_crop", return_value=True)

        def fake_enqueue_prediciton(obj, x, idx, result_handler):
            result_handler(idx, self.fake_ann_scene, (x, None, None, 0))

        mock_enqueue = mocker.patch.object(
            ClassificationOpenVINOInferencer, "enqueue_prediction", fake_enqueue_prediciton
        )

        updated_dataset = self.cls_ov_task.infer(
            self.dataset, InferenceParameters(enable_async_inference=True, is_evaluation=True)
        )

        for updated in updated_dataset:
            assert updated.annotation_scene.contains_any(self.labels)

    @e2e_pytest_unit
    def test_explain(self, mocker):
        self.fake_silency_map = np.random.randint(255, size=(2, 224, 224), dtype=np.uint8)
        mocker.patch.object(
            ClassificationOpenVINOInferencer,
            "predict",
            return_value=(
                self.fake_ann_scene,
                np.array([0, 1]),
                self.fake_silency_map,
                self.fake_input,
                self.fake_input,
            ),
        )
        self.cls_ov_task.inferencer.model.hierarchical = False
        updpated_dataset = self.cls_ov_task.explain(self.dataset)

        assert updpated_dataset is not None
        assert updpated_dataset.get_labels() == self.dataset.get_labels()

    @e2e_pytest_unit
    def test_evaluate(self, mocker):
        result_set = ResultSetEntity(
            model=None,
            ground_truth_dataset=DatasetEntity(),
            prediction_dataset=DatasetEntity(),
        )
        fake_metrics = mocker.patch("otx.api.usecases.evaluation.accuracy.Accuracy", autospec=True)
        fake_metrics.get_performance.return_value = Performance(
            score=ScoreMetric(name="fake", value=0.1), dashboard_metrics="Accuracy"
        )
        mocker.patch.object(MetricsHelper, "compute_accuracy", return_value=fake_metrics)
        self.cls_ov_task.evaluate(result_set)

        assert result_set.performance.score.value == 0.1

    @e2e_pytest_unit
    def test_deploy(self, otx_model):
        output_model = copy.deepcopy(otx_model)
        self.cls_ov_task.model.set_data("openvino.bin", b"foo")
        self.cls_ov_task.model.set_data("openvino.xml", b"bar")
        self.cls_ov_task.deploy(output_model)

        assert output_model.exportable_code is not None

    @e2e_pytest_unit
    def test_optimize(self, mocker, otx_model):
        def patch_save_model(model, output_xml):
            with open(output_xml, "wb") as f:
                f.write(b"foo")
            bin_path = pathlib.Path(output_xml).parent / pathlib.Path(str(pathlib.Path(output_xml).stem) + ".bin")
            with open(bin_path, "wb") as f:
                f.write(b"bar")

        output_model = copy.deepcopy(otx_model)
        self.cls_ov_task.model.set_data("openvino.bin", b"foo")
        self.cls_ov_task.model.set_data("openvino.xml", b"bar")
        mocker.patch("otx.algorithms.classification.adapters.openvino.task.ov.Core.read_model", autospec=True)
        mocker.patch("otx.algorithms.classification.adapters.openvino.task.ov.serialize", new=patch_save_model)
        fake_quantize = mocker.patch(
            "otx.algorithms.classification.adapters.openvino.task.nncf.quantize", autospec=True
        )
        self.cls_ov_task.optimize(OptimizationType.POT, dataset=self.dataset, output_model=output_model)

        fake_quantize.assert_called_once()
        assert self.cls_ov_task.model.get_data("openvino.bin")
        assert self.cls_ov_task.model.get_data("openvino.xml")
