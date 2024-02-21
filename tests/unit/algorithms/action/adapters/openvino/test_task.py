"""Unit Test for otx.algorithms.action.adapters.openvino.task."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
import pathlib
from typing import Any, Dict

import numpy as np
import pytest
from openvino.model_api.adapters import OpenvinoAdapter

from otx.algorithms.action.adapters.openvino import ActionOVClsDataLoader
from otx.algorithms.action.configs.base.configuration import ActionConfig
from otx.algorithms.action.adapters.openvino.task import (
    ActionOpenVINOInferencer,
    ActionOpenVINOTask,
    DataLoaderWrapper,
)
from otx.api.configuration import ConfigurableParameters
from otx.api.entities.annotation import (
    Annotation,
    AnnotationSceneEntity,
    AnnotationSceneKind,
)
from otx.api.entities.label import Domain
from otx.api.entities.label_schema import LabelGroup, LabelGroupType, LabelSchemaEntity
from otx.api.entities.model import (
    ModelConfiguration,
    ModelEntity,
    ModelFormat,
    ModelOptimizationType,
    ModelPrecision,
    OptimizationMethod,
)
from otx.api.entities.model_template import InstantiationType, TaskFamily, TaskType
from otx.api.entities.optimization_parameters import OptimizationParameters
from otx.api.entities.resultset import ResultSetEntity
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.shapes.rectangle import Rectangle
from otx.api.entities.task_environment import TaskEnvironment
from otx.api.usecases.evaluation.metrics_helper import MetricsHelper
from otx.api.usecases.exportable_code.inference import IInferencer
from otx.api.usecases.exportable_code.prediction_to_annotation_converter import (
    ClassificationToAnnotationConverter,
    DetectionBoxToAnnotationConverter,
)
from otx.api.usecases.tasks.interfaces.optimization_interface import OptimizationType
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.action.test_helpers import (
    MockModelTemplate,
    generate_action_cls_otx_dataset,
    generate_labels,
    return_args,
)


class MockOVInferencer(IInferencer):
    """Mock class for OV inferencer."""

    def __init__(self, *args, **kwargs):
        self.model = MockModel()
        self.model.t = 8
        self.model.w = 256
        self.model.h = 256
        self.labels = generate_labels(1, Domain.ACTION_CLASSIFICATION)
        self.configuration: Dict[Any, Any] = {}

    def predict(self, data):
        return AnnotationSceneEntity(
            annotations=[Annotation(shape=Rectangle(0, 0, 1, 1), labels=[ScoredLabel(self.labels[0], 1.0)])],
            kind=AnnotationSceneKind.PREDICTION,
        )

    def pre_process(self, item):
        return item, {"dummy_meta": "dummy_info"}

    def forward(self, item):
        pass

    def post_process(self, item):
        pass


class MockModel:
    """Mock class for OV model."""

    def preprocess(self, image):
        return "Preprocess function is called", None

    def postprocess(self, prediction, metadata):
        return "Postprocess function is called"

    def infer_sync(self, image):
        return "Funtion infer_sync is called"


class MockOpenvinoAdapter(OpenvinoAdapter):
    """Mock class for OpenvinoAdapter."""

    def __init__(self, *args, **kwargs):
        pass


class MockDataloader(ActionOVClsDataLoader):
    """Mock class for dataloader for OpenVINO inference."""

    def __init__(self, dataset, *args, **kwargs):
        self.dataset = dataset

    def __len__(self):
        return 1

    def __getitem__(self, index):
        if index >= len(self):
            raise StopIteration
        return self.dataset._items

    def add_prediction(self, dataset, data, prediction):
        for dataset_item in dataset:
            dataset_item.append_labels(prediction.annotations[0].get_labels())


class TestActionOVInferencer:
    """Test class for ActionOpenVINOInferencer."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        self.labels = generate_labels(3, Domain.ACTION_CLASSIFICATION)
        self.label_schema = LabelSchemaEntity()
        label_group = LabelGroup(
            name="labels",
            labels=self.labels,
            group_type=LabelGroupType.EXCLUSIVE,
        )
        self.label_schema.add_group(label_group)
        mocker.patch("otx.algorithms.action.adapters.openvino.task.OpenvinoAdapter.__init__", return_value=None)
        mocker.patch("otx.algorithms.action.adapters.openvino.task.Model.create_model", return_value=MockModel())
        self.inferencer = ActionOpenVINOInferencer(
            "ACTION_CLASSIFICATION",
            ActionConfig(),
            self.label_schema,
            "openvino.xml",
            "openvino.bin",
        )

    @e2e_pytest_unit
    def test_init(self) -> None:
        """Test __init__ function."""
        inferencer = ActionOpenVINOInferencer(
            "ACTION_CLASSIFICATION",
            ActionConfig(),
            self.label_schema,
            "openvino.xml",
            "openvino.bin",
        )
        assert inferencer.task_type == "ACTION_CLASSIFICATION"
        assert inferencer.label_schema == self.label_schema
        assert isinstance(inferencer.model, MockModel)
        assert isinstance(inferencer.converter, ClassificationToAnnotationConverter)

        inferencer = ActionOpenVINOInferencer(
            "ACTION_DETECTION",
            ActionConfig(),
            self.label_schema,
            "openvino.xml",
            "openvino.bin",
        )
        assert inferencer.task_type == "ACTION_DETECTION"
        assert isinstance(inferencer.converter, DetectionBoxToAnnotationConverter)

    @e2e_pytest_unit
    def test_pre_process(self) -> None:
        """Test pre_process funciton."""
        dataset = generate_action_cls_otx_dataset(1, 10, self.labels)
        inputs = dataset._items
        assert self.inferencer.pre_process(inputs) == ("Preprocess function is called", None)

    @e2e_pytest_unit
    def test_post_process(self, mocker) -> None:
        """Test post_process function."""
        mocker.patch(
            "otx.algorithms.action.adapters.openvino.task.ClassificationToAnnotationConverter.convert_to_annotation",
            side_effect=return_args,
        )
        assert (
            self.inferencer.post_process({"dummy": np.ndarray(1)}, {"dummy": "meta"})[0][0]
            == "Postprocess function is called"
        )

    @e2e_pytest_unit
    def test_forward(self) -> None:
        """Test forward function."""

        assert self.inferencer.forward({"dummy": np.ndarray(1)}) == "Funtion infer_sync is called"

    @e2e_pytest_unit
    def test_predict(self, mocker) -> None:
        """Test predict function."""

        mocker.patch(
            "otx.algorithms.action.adapters.openvino.task.ActionOpenVINOInferencer.pre_process",
            return_value=("data", "metadata"),
        )
        mocker.patch(
            "otx.algorithms.action.adapters.openvino.task.ActionOpenVINOInferencer.forward",
            return_value="raw_predictions",
        )
        mocker.patch(
            "otx.algorithms.action.adapters.openvino.task.ActionOpenVINOInferencer.post_process",
            return_value="predictions",
        )

        dataset = generate_action_cls_otx_dataset(1, 10, self.labels)
        inputs = dataset._items
        assert self.inferencer.predict(inputs) == "predictions"


class TestActionOVTask:
    """Test class for ActionOpenVINOTask."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        self.video_len = 1
        self.frame_len = 10

        labels = generate_labels(3, Domain.ACTION_CLASSIFICATION)
        self.label_schema = LabelSchemaEntity()
        label_group = LabelGroup(
            name="labels",
            labels=labels,
            group_type=LabelGroupType.EXCLUSIVE,
        )
        self.label_schema.add_group(label_group)
        template = MockModelTemplate(
            model_template_id="template_id",
            model_template_path="template_path",
            name="template",
            task_family=TaskFamily.VISION,
            task_type=TaskType.ACTION_CLASSIFICATION,
            instantiation=InstantiationType.CLASS,
        )

        config = ModelConfiguration(ActionConfig(), self.label_schema)
        self.dataset = generate_action_cls_otx_dataset(1, 10, labels)
        self.model = ModelEntity(self.dataset, config)
        self.model.set_data("openvino.xml", np.ndarray([1]).tobytes())
        self.model.set_data("openvino.bin", np.ndarray([1]).tobytes())

        self.task_environment = TaskEnvironment(
            model=self.model,
            hyper_parameters=ConfigurableParameters(header="h-params"),
            label_schema=self.label_schema,
            model_template=template,
        )

    @e2e_pytest_unit
    def test_load_inferencer(self, mocker) -> None:
        """Test load_inferencer function."""

        mocker.patch(
            "otx.algorithms.action.adapters.openvino.task.ActionOpenVINOInferencer", return_value=MockOVInferencer()
        )
        task = ActionOpenVINOTask(self.task_environment)
        assert isinstance(task.inferencer, MockOVInferencer)

        self.task_environment.model = None
        with pytest.raises(RuntimeError):
            task = ActionOpenVINOTask(self.task_environment)

    @e2e_pytest_unit
    def test_infer(self, mocker) -> None:
        """Test infer function."""

        mocker.patch(
            "otx.algorithms.action.adapters.openvino.task.ActionOpenVINOTask.load_inferencer",
            return_value=MockOVInferencer(),
        )
        mocker.patch(
            "otx.algorithms.action.adapters.openvino.task.get_ovdataloader", return_value=MockDataloader(self.dataset)
        )
        task = ActionOpenVINOTask(self.task_environment)
        output = task.infer(self.dataset.with_empty_annotations())
        assert output[0].annotation_scene.kind == AnnotationSceneKind.PREDICTION

    @e2e_pytest_unit
    def test_evaluate(self, mocker) -> None:
        """Test evaluate function."""

        class MockPerformance:
            def get_performance(self):
                return 1.0

        mocker.patch(
            "otx.algorithms.action.adapters.openvino.task.ActionOpenVINOTask.load_inferencer",
            return_value=MockOVInferencer(),
        )
        task = ActionOpenVINOTask(self.task_environment)

        resultset = ResultSetEntity(
            self.model,
            self.dataset,
            self.dataset,
        )

        mocker.patch.object(MetricsHelper, "compute_accuracy", return_value=MockPerformance())
        task.evaluate(resultset, "Accuracy")
        assert resultset.performance == 1.0

        mocker.patch.object(MetricsHelper, "compute_f_measure", return_value=MockPerformance())
        self.task_environment.model_template.task_type = TaskType.ACTION_DETECTION
        task = ActionOpenVINOTask(self.task_environment)
        task.evaluate(resultset, "Accuracy")
        assert resultset.performance == 1.0

    @e2e_pytest_unit
    def test_deploy(self, mocker) -> None:
        """Test function for deploy function."""

        mocker.patch(
            "otx.algorithms.action.adapters.openvino.task.ActionOpenVINOTask.load_inferencer",
            return_value=MockOVInferencer(),
        )
        task = ActionOpenVINOTask(self.task_environment)
        assert self.model.exportable_code is None
        task.deploy(self.model)
        assert self.model.exportable_code is not None

    @e2e_pytest_unit
    def test_optimize(self, mocker) -> None:
        """Test optimization function."""

        class MockPipeline:
            """Mock class for POT pipeline"""

            def run(self, model):
                return model

        def mock_save_model(model, output_xml):
            """Mock function for save_model function."""
            with open(output_xml, "wb") as f:
                f.write(np.ndarray(1).tobytes())
            bin_path = pathlib.Path(output_xml).parent / pathlib.Path(str(pathlib.Path(output_xml).stem) + ".bin")
            with open(bin_path, "wb") as f:
                f.write(np.ndarray(1).tobytes())

        mocker.patch("otx.algorithms.action.adapters.openvino.task.ov.Core.read_model", autospec=True)
        mocker.patch("otx.algorithms.action.adapters.openvino.task.ov.save_model", new=mock_save_model)
        fake_quantize = mocker.patch("otx.algorithms.action.adapters.openvino.task.nncf.quantize", autospec=True)

        mocker.patch(
            "otx.algorithms.action.adapters.openvino.task.get_ovdataloader", return_value=MockDataloader(self.dataset)
        )
        mocker.patch(
            "otx.algorithms.action.adapters.openvino.task.DataLoaderWrapper", return_value=MockDataloader(self.dataset)
        )
        mocker.patch(
            "otx.algorithms.action.adapters.openvino.task.ActionOpenVINOTask.load_inferencer",
            return_value=MockOVInferencer(),
        )
        task = ActionOpenVINOTask(self.task_environment)
        task.optimize(OptimizationType.POT, self.dataset, self.model, OptimizationParameters())
        fake_quantize.assert_called_once()
        assert self.model.get_data("openvino.xml") is not None
        assert self.model.get_data("openvino.bin") is not None
        assert self.model.model_format == ModelFormat.OPENVINO
        assert self.model.optimization_type == ModelOptimizationType.POT
        assert self.model.optimization_methods == [OptimizationMethod.QUANTIZATION]
        assert self.model.precision == [ModelPrecision.INT8]


class TestDataLoaderWrapper:
    """Test class for DataLoaderWrapper"""

    def setup(self, mocker) -> None:
        labels = generate_labels(3, Domain.ACTION_CLASSIFICATION)
        self.dataset = generate_action_cls_otx_dataset(1, 10, labels)
        ovdataloader = MockDataloader(self.dataset)
        inferencer = MockOVInferencer()
        self.dataloader = DataLoaderWrapper(ovdataloader, inferencer)

    @e2e_pytest_unit
    def test_len(self) -> None:
        """Test __len__ function."""

        assert len(self.dataloader) == 1

    def test_getitem(self) -> None:
        """Test __getitem__ function."""

        out = self.dataloader[0]
        assert isinstance(out[1], AnnotationSceneEntity)
        assert len(out[0]) == 29
