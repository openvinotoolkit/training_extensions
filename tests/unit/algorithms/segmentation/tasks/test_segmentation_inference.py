# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os

import numpy as np
import pytest

from otx.algorithms.common.tasks import BaseTask
from otx.algorithms.segmentation.tasks import SegmentationInferenceTask
from otx.api.configuration.helper import create
from otx.api.entities.annotation import Annotation
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.label import LabelEntity
from otx.api.entities.metrics import Performance, ScoreMetric
from otx.api.entities.model import ModelPrecision
from otx.api.entities.model_template import parse_model_template
from otx.api.entities.resultset import ResultSetEntity
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.shapes.polygon import Point, Polygon
from otx.api.usecases.evaluation.metrics_helper import MetricsHelper
from otx.api.usecases.tasks.interfaces.export_interface import ExportType
from otx.api.utils.shape_factory import ShapeFactory
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.segmentation.test_helpers import (
    DEFAULT_SEG_TEMPLATE_DIR,
    generate_otx_dataset,
    init_environment,
)


class TestOTXSegTaskInference:
    @pytest.fixture(autouse=True)
    def setup(self, otx_model, tmp_dir_path) -> None:
        model_template = parse_model_template(os.path.join(DEFAULT_SEG_TEMPLATE_DIR, "template.yaml"))
        hyper_parameters = create(model_template.hyper_parameters.data)
        task_env = init_environment(hyper_parameters, model_template)
        self.output_path = str(tmp_dir_path)
        self.seg_train_task = SegmentationInferenceTask(task_env, output_path=self.output_path)
        self.model = otx_model

    @e2e_pytest_unit
    def test_infer(self, mocker):
        dataset = generate_otx_dataset(5)
        fake_output = {"outputs": {"eval_predictions": np.zeros((5, 1)), "feature_vectors": np.zeros((5, 1))}}
        fake_annotation = [
            Annotation(
                Polygon(points=[Point(0, 0)]),
                id=0,
                labels=[ScoredLabel(LabelEntity(name="fake", domain="SEGMENTATION"), probability=1.0)],
            )
        ]

        mock_run_task = mocker.patch.object(BaseTask, "_run_task", return_value=fake_output)
        mocker.patch("numpy.transpose")
        mocker.patch("otx.algorithms.segmentation.tasks.inference.create_hard_prediction_from_soft_prediction")
        mocker.patch(
            "otx.algorithms.segmentation.tasks.inference.create_annotation_from_segmentation_map",
            return_value=fake_annotation,
        )
        mocker.patch("otx.algorithms.segmentation.tasks.inference.get_activation_map", return_value=np.zeros((1, 1)))
        mocker.patch.object(ShapeFactory, "shape_produces_valid_crop", return_value=True)

        updated_dataset = self.seg_train_task.infer(dataset, None)

        mock_run_task.assert_called_once()
        for updated in updated_dataset:
            assert updated.annotation_scene.contains_any([LabelEntity(name="fake", domain="SEGMENTATION")])

    @e2e_pytest_unit
    def test_evaluate(self, mocker):
        result_set = ResultSetEntity(
            model=self.model,
            ground_truth_dataset=DatasetEntity(),
            prediction_dataset=DatasetEntity(),
        )
        fake_metrics = mocker.patch("otx.api.usecases.evaluation.dice.DiceAverage", autospec=True)
        fake_metrics.get_performance.return_value = Performance(
            score=ScoreMetric(name="fake", value=0.1), dashboard_metrics="mDice"
        )
        mocker.patch.object(MetricsHelper, "compute_dice_averaged_over_pixels", return_value=fake_metrics)
        self.seg_train_task.evaluate(result_set)

        assert result_set.performance.score.value == 0.1

    @pytest.mark.parametrize("precision", [ModelPrecision.FP16, ModelPrecision.FP32])
    @e2e_pytest_unit
    def test_export(self, mocker, precision: ModelPrecision):
        fake_output = {"outputs": {"bin": None, "xml": None}}
        mock_run_task = mocker.patch.object(BaseTask, "_run_task", return_value=fake_output)

        with pytest.raises(RuntimeError):
            self.seg_train_task.export(ExportType.OPENVINO, self.model, precision)
            mock_run_task.assert_called_once()

    @pytest.mark.parametrize("precision", [ModelPrecision.FP16, ModelPrecision.FP32])
    @e2e_pytest_unit
    def test_export_with_model_files(self, mocker, precision: ModelPrecision):
        with open(f"{self.output_path}/model.xml", "wb") as f:
            f.write(b"foo")
        with open(f"{self.output_path}/model.bin", "wb") as f:
            f.write(b"bar")

        fake_output = {"outputs": {"bin": f"{self.output_path}/model.xml", "xml": f"{self.output_path}/model.bin"}}
        mock_run_task = mocker.patch.object(BaseTask, "_run_task", return_value=fake_output)
        self.seg_train_task.export(ExportType.OPENVINO, self.model, precision)

        mock_run_task.assert_called_once()
        assert self.model.get_data("openvino.bin")
        assert self.model.get_data("openvino.xml")

    @e2e_pytest_unit
    def test_unload(self, mocker):
        mock_cleanup = mocker.patch.object(BaseTask, "cleanup")
        self.seg_train_task.unload()

        mock_cleanup.assert_called_once()

    @e2e_pytest_unit
    @pytest.mark.parametrize("model_dir", ["...", ".../supcon", ".../supcon/workspace"])
    def test_init_recipe_supcon(self, mocker, model_dir: str):
        mocker.patch("otx.algorithms.segmentation.tasks.inference.SegmentationInferenceTask._init_model_cfg")
        mocker.patch("otx.algorithms.segmentation.tasks.inference.patch_default_config")
        mocker.patch("otx.algorithms.segmentation.tasks.inference.patch_runner")
        mocker.patch("otx.algorithms.segmentation.tasks.inference.patch_data_pipeline")
        mocker.patch("otx.algorithms.segmentation.tasks.inference.patch_datasets")
        mocker.patch("otx.algorithms.segmentation.tasks.inference.patch_evaluation")

        self.seg_train_task._hyperparams.learning_parameters.enable_supcon = True
        self.seg_train_task._model_dir = model_dir

        self.seg_train_task._init_recipe()

        assert self.seg_train_task._model_dir.endswith("supcon")

        self.seg_train_task._hyperparams.learning_parameters.enable_supcon = False
        self.seg_train_task._hyperparams._model_dir = os.path.abspath(DEFAULT_SEG_TEMPLATE_DIR)
