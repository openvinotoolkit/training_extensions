"""Tests the methods in the OpenVINO task."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import torch
import pytest
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.algorithms.visual_prompting.adapters.pytorch_lightning.datasets.dataset import OTXVisualPromptingDataset
from openvino.model_api.models import Model
from otx.algorithms.visual_prompting.configs.base import VisualPromptingBaseConfig
from otx.algorithms.visual_prompting.tasks.openvino import (
    OpenVINOVisualPromptingInferencer,
    OpenVINOVisualPromptingTask,
)
from otx.api.entities.annotation import (
    Annotation,
    AnnotationSceneEntity,
    AnnotationSceneKind,
)
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.inference_parameters import InferenceParameters
from otx.api.entities.label import LabelEntity
from otx.api.entities.metrics import Performance, ScoreMetric
from otx.api.entities.resultset import ResultSetEntity
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.shapes.polygon import Point, Polygon
from otx.api.usecases.evaluation.metrics_helper import MetricsHelper
from otx.api.utils.shape_factory import ShapeFactory

from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.visual_prompting.test_helpers import (
    generate_visual_prompting_dataset,
    init_environment,
)
from otx.api.usecases.exportable_code.prediction_to_annotation_converter import (
    VisualPromptingToAnnotationConverter,
)


class TestOpenVINOVisualPromptingInferencer:
    @pytest.fixture(autouse=True)
    def setup(self, mocker):
        self.fake_annotation = [
            Annotation(
                Polygon(points=[Point(0, 0)]),
                id=0,
                labels=[ScoredLabel(LabelEntity(name="fake", domain="VISUALPROMPTING"), probability=1.0)],
            )
        ]
        mocker.patch("otx.algorithms.visual_prompting.tasks.openvino.OpenvinoAdapter")
        mocker.patch.object(Model, "create_model")
        mocker.patch.object(
            VisualPromptingToAnnotationConverter, "convert_to_annotation", return_value=self.fake_annotation
        )
        self.task_environment = init_environment()
        visual_prompting_hparams = self.task_environment.get_hyper_parameters(VisualPromptingBaseConfig)
        label_schema = self.task_environment.label_schema

        self.visual_prompting_ov_inferencer = OpenVINOVisualPromptingInferencer(
            visual_prompting_hparams,
            label_schema,
            {"image_encoder": "", "decoder": ""},
            {"image_encoder": "", "decoder": ""},
        )
        self.visual_prompting_ov_inferencer.model["decoder"] = mocker.patch(
            "openvino.model_api.models.Model", autospec=True
        )

    @e2e_pytest_unit
    def test_pre_process(self, mocker):
        """Test pre_process."""
        mocker_get_prompts = mocker.patch.object(OTXVisualPromptingDataset, "get_prompts", return_value={})
        mocker.patch.object(self.visual_prompting_ov_inferencer, "transform", lambda items: items)
        fake_input = mocker.Mock(spec=DatasetItemEntity)

        returned_value = self.visual_prompting_ov_inferencer.pre_process(fake_input)

        assert "index" in returned_value
        assert returned_value.get("index") == 0
        assert "images" in returned_value
        mocker_get_prompts.assert_called_once()

    @e2e_pytest_unit
    def test_post_process(self, mocker):
        """Test post_process."""
        fake_prediction = {"masks": np.empty((1, 1, 2, 2))}
        fake_metadata = {"label": mocker.Mock(spec=LabelEntity), "original_size": np.array((2, 2))}
        self.visual_prompting_ov_inferencer.model["decoder"].postprocess.return_value = (
            np.ones((2, 2)),
            np.ones((2, 2)),
        )

        returned_value = self.visual_prompting_ov_inferencer.post_process(fake_prediction, fake_metadata)

        assert len(returned_value) == 3
        assert np.array_equal(returned_value[0], self.fake_annotation)
        assert np.array_equal(returned_value[1], np.ones((2, 2)))
        assert np.array_equal(returned_value[2], np.ones((2, 2)))

    @e2e_pytest_unit
    def test_predict(self, mocker):
        """Teset predict."""
        mocker_pre_process = mocker.patch.object(
            OpenVINOVisualPromptingInferencer,
            "pre_process",
            return_value={
                "index": 0,
                "images": torch.rand((1, 3, 2, 2)),
                "bboxes": [np.array([[[1, 1], [2, 2]]])],
                "labels": [1, 2],
                "original_size": (4, 4),
            },
        )
        mocker_pre_process_decoder = mocker.patch.object(
            self.visual_prompting_ov_inferencer.model["decoder"], "preprocess", return_value={}
        )
        mocker_forward = mocker.patch.object(
            OpenVINOVisualPromptingInferencer, "forward", return_value={"image_embeddings": np.empty((4, 2, 2))}
        )
        mocker_forward_decoder = mocker.patch.object(
            OpenVINOVisualPromptingInferencer, "forward_decoder", return_value=None
        )
        mocker_post_process = mocker.patch.object(
            OpenVINOVisualPromptingInferencer, "post_process", return_value=(self.fake_annotation, None, None)
        )
        fake_input = mocker.Mock(spec=DatasetItemEntity)

        returned_value = self.visual_prompting_ov_inferencer.predict(fake_input)

        mocker_pre_process.assert_called_once()
        mocker_pre_process_decoder.assert_called_once()
        mocker_forward.assert_called_once()
        mocker_forward_decoder.assert_called_once()
        mocker_post_process.assert_called_once()
        assert returned_value == self.fake_annotation

    @e2e_pytest_unit
    def test_forward(self):
        """Test forward."""
        fake_input = {"images": np.ones((1, 3, 2, 2))}
        fake_output = {"image_embeddings": np.ones((1, 1, 2, 2))}
        self.visual_prompting_ov_inferencer.model["image_encoder"].infer_sync.return_value = fake_output
        returned_value = self.visual_prompting_ov_inferencer.forward(fake_input)

        assert returned_value == fake_output

    @e2e_pytest_unit
    def test_forward_decoder(self):
        """Test forward_decoder."""
        fake_input = {}
        fake_output = {"masks": np.ones((1, 1, 2, 2))}
        self.visual_prompting_ov_inferencer.model["decoder"].infer_sync.return_value = fake_output
        returned_value = self.visual_prompting_ov_inferencer.forward_decoder(fake_input)

        assert returned_value == fake_output


class TestOpenVINOVisualPromptingTask:
    @pytest.fixture(autouse=True)
    def setup(self, mocker):
        """Load the OpenVINOVisualPromptingTask."""
        mocker.patch("otx.algorithms.visual_prompting.tasks.openvino.OpenvinoAdapter")
        mocker.patch.object(Model, "create_model")
        self.task_environment = init_environment()
        visual_prompting_hparams = self.task_environment.get_hyper_parameters(VisualPromptingBaseConfig)

        visual_prompting_ov_inferencer = OpenVINOVisualPromptingInferencer(
            visual_prompting_hparams,
            self.task_environment.label_schema,
            {"image_encoder": "", "decoder": ""},
            {"image_encoder": "", "decoder": ""},
        )

        self.task_environment.model = mocker.patch("otx.api.entities.model.ModelEntity")
        mocker.patch.object(OpenVINOVisualPromptingTask, "load_inferencer", return_value=visual_prompting_ov_inferencer)
        self.visual_prompting_ov_task = OpenVINOVisualPromptingTask(task_environment=self.task_environment)

    @e2e_pytest_unit
    def test_infer(self, mocker):
        """Test infer."""
        fake_annotation = [
            Annotation(
                Polygon(points=[Point(0, 0)]),
                id=0,
                labels=[ScoredLabel(LabelEntity(name="fake", domain="VISUALPROMPTING"), probability=1.0)],
            )
        ]

        mocker_predict = mocker.patch.object(OpenVINOVisualPromptingInferencer, "predict", return_value=fake_annotation)
        mocker.patch.object(ShapeFactory, "shape_produces_valid_crop", return_value=True)

        dataset = generate_visual_prompting_dataset()

        updated_dataset = self.visual_prompting_ov_task.infer(
            dataset, InferenceParameters(enable_async_inference=False)
        )

        for updated in updated_dataset:
            assert updated.annotation_scene.contains_any([LabelEntity(name="fake", domain="VISUALPROMPTING")])

        mocker_predict.assert_called()
        assert mocker_predict.call_count == len(updated_dataset)

    @e2e_pytest_unit
    def test_evaluate(self, mocker):
        """Test evaluate."""
        result_set = ResultSetEntity(
            model=None,
            ground_truth_dataset=DatasetEntity(),
            prediction_dataset=DatasetEntity(),
        )
        fake_metrics = mocker.patch("otx.api.usecases.evaluation.dice.DiceAverage", autospec=True)
        fake_metrics.get_performance.return_value = Performance(
            score=ScoreMetric(name="fake", value=0.1), dashboard_metrics="mDice"
        )
        mocker.patch.object(MetricsHelper, "compute_dice_averaged_over_pixels", return_value=fake_metrics)
        self.visual_prompting_ov_task.evaluate(result_set)

        assert result_set.performance.score.value == 0.1
