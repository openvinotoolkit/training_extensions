# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pytest

from otx.algorithms.classification.tasks import ClassificationInferenceTask
from otx.algorithms.common.tasks import BaseTask
from otx.api.configuration.configurable_parameters import ConfigurableParameters
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.inference_parameters import InferenceParameters
from otx.api.entities.label import Domain
from otx.api.entities.label_schema import LabelSchemaEntity
from otx.api.entities.metrics import Performance, ScoreMetric
from otx.api.entities.model import ModelConfiguration, ModelEntity, ModelPrecision
from otx.api.entities.resultset import ResultSetEntity
from otx.api.usecases.evaluation.metrics_helper import MetricsHelper
from otx.api.usecases.tasks.interfaces.export_interface import ExportType
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.classification.test_helper import (
    DEFAULT_CLS_TEMPLATE,
    init_environment,
    setup_configurable_parameters,
)


@pytest.fixture
def otx_classification_model():
    model_configuration = ModelConfiguration(
        configurable_parameters=ConfigurableParameters(header="header", description="description"),
        label_schema=LabelSchemaEntity(),
    )
    return ModelEntity(train_dataset=DatasetEntity(), configuration=model_configuration)


class TestOTXClassificationInferenceTask:
    @pytest.fixture(autouse=True)
    def setup(self, otx_classification_model, tmp_dir_path) -> None:
        self.dataset_len = 5
        self.hyper_parameters, self.model_template = setup_configurable_parameters(DEFAULT_CLS_TEMPLATE)
        task_environment, self.dataset = init_environment(
            self.hyper_parameters, self.model_template, False, False, self.dataset_len
        )

        self.output_path = str(tmp_dir_path)
        self.cls_inference_task = ClassificationInferenceTask(task_environment, output_path=self.output_path)
        self.model = otx_classification_model

    @pytest.mark.parametrize("multilabel, hierarchical", [(False, False), (True, False), (False, True)])
    @e2e_pytest_unit
    def test_infer(self, mocker, multilabel, hierarchical):
        task_environment, dataset = init_environment(
            self.hyper_parameters, self.model_template, multilabel, hierarchical, self.dataset_len
        )
        custom_cls_inference_task = ClassificationInferenceTask(task_environment, output_path=self.output_path)

        items_num = len(dataset)
        fake_output = {
            "outputs": {
                "eval_predictions": np.zeros((items_num, 5)),
                "feature_vectors": np.zeros((items_num, 5)),
                "saliency_maps": np.zeros((items_num, 5, 5)).astype(np.uint8),
            }
        }

        mock_run_task = mocker.patch.object(BaseTask, "_run_task", return_value=fake_output)
        inf_params = InferenceParameters()
        inf_params.is_evaluation = True
        updated_dataset = custom_cls_inference_task.infer(dataset.with_empty_annotations(), inf_params)

        mock_run_task.assert_called_once()
        for updated in updated_dataset:
            if not multilabel and not hierarchical:
                assert len(updated.annotation_scene.get_labels()) == 1
            for lbl in updated.annotation_scene.get_labels():
                assert lbl.domain == Domain.CLASSIFICATION

    @e2e_pytest_unit
    def test_explain(self, mocker):
        items_num = len(self.dataset)
        fake_output = {
            "outputs": {
                "eval_predictions": np.zeros((items_num, 5)),
                "feature_vectors": np.zeros((items_num, 5)),
                "saliency_maps": np.zeros((items_num, 5, 5)).astype(np.uint8),
            }
        }

        mock_run_task = mocker.patch.object(BaseTask, "_run_task", return_value=fake_output)
        updated_dataset = self.cls_inference_task.explain(self.dataset.with_empty_annotations())
        mock_run_task.assert_called_once()

        for item in updated_dataset:
            data_list = item.get_metadata()
            for data in data_list:
                assert data.data.type == "saliency_map"
                assert data.data.numpy is not None

    @e2e_pytest_unit
    def test_evaluate(self, mocker):
        result_set = ResultSetEntity(
            model=self.model,
            ground_truth_dataset=DatasetEntity(),
            prediction_dataset=DatasetEntity(),
        )
        fake_metrics = mocker.patch("otx.api.usecases.evaluation.accuracy.Accuracy", autospec=True)
        fake_metrics.get_performance.return_value = Performance(
            score=ScoreMetric(name="fake", value=0.1), dashboard_metrics="accuracy"
        )
        mocker.patch.object(MetricsHelper, "compute_accuracy", return_value=fake_metrics)
        self.cls_inference_task.evaluate(result_set)

        assert result_set.performance.score.value == 0.1

    @pytest.mark.parametrize("precision", [ModelPrecision.FP16, ModelPrecision.FP32])
    @e2e_pytest_unit
    def test_export_empty_model(self, mocker, precision: ModelPrecision):
        fake_output = {"outputs": {"bin": None, "xml": None}}
        mock_run_task = mocker.patch.object(BaseTask, "_run_task", return_value=fake_output)

        with pytest.raises(RuntimeError):
            self.cls_inference_task.export(ExportType.OPENVINO, self.model, precision)
            mock_run_task.assert_called_once()

    @pytest.mark.parametrize("precision", [ModelPrecision.FP16, ModelPrecision.FP32])
    @e2e_pytest_unit
    def test_export_with_model_files(self, mocker, precision: ModelPrecision):
        with open(f"{self.output_path}/model.xml", "wb") as f:
            f.write(b"foo")
        with open(f"{self.output_path}/model.bin", "wb") as f:
            f.write(b"bar")

        mocker.patch("otx.algorithms.classification.tasks.inference.embed_ir_model_data", return_value=None)
        fake_output = {"outputs": {"bin": f"{self.output_path}/model.xml", "xml": f"{self.output_path}/model.bin"}}
        mock_run_task = mocker.patch.object(BaseTask, "_run_task", return_value=fake_output)
        self.cls_inference_task.export(ExportType.OPENVINO, self.model, precision)

        mock_run_task.assert_called_once()
        assert self.model.get_data("openvino.bin")
        assert self.model.get_data("openvino.xml")

    @e2e_pytest_unit
    def test_unload(self, mocker):
        mock_cleanup = mocker.patch.object(BaseTask, "cleanup")
        self.cls_inference_task.unload()

        mock_cleanup.assert_called_once()
