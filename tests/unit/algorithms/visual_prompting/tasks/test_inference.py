"""Tests the methods in the inference task."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from collections import OrderedDict

import pytest
from omegaconf import DictConfig

from otx.algorithms.visual_prompting.tasks.inference import InferenceTask
from otx.api.entities.metrics import NullPerformance
from otx.api.entities.model import ModelEntity
from otx.api.entities.resultset import ResultSetEntity
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.visual_prompting.test_helpers import (
    generate_visual_prompting_dataset,
    init_environment,
)


class TestInferenceTask:
    @pytest.fixture(autouse=True)
    def setup(self, tmpdir):
        self.task_environment = init_environment()
        self.output_path = str(tmpdir.mkdir("visual_prompting_training_test"))
        self.inference_task = InferenceTask(self.task_environment, self.output_path)

    @e2e_pytest_unit
    def test_get_config(self):
        """Test get_config."""
        assert isinstance(self.inference_task.config, DictConfig)
        assert self.inference_task.config.dataset.task == "visual_prompting"

    @e2e_pytest_unit
    def test_load_model_without_otx_model(self, mocker):
        """Test load_model without otx_model."""
        mocker_segment_anything = mocker.patch(
            "otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.SegmentAnything"
        )
        otx_model = None

        self.inference_task.load_model(otx_model)

        mocker_segment_anything.assert_called_once()

    @e2e_pytest_unit
    def test_load_model_with_otx_model(self, mocker):
        """Test load_model with otx_model."""
        mocker_segment_anything = mocker.patch(
            "otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.SegmentAnything"
        )
        mocker_otx_model = mocker.patch("otx.api.entities.model.ModelEntity")
        mocker_io_bytes_io = mocker.patch("io.BytesIO")
        mocker_torch_load = mocker.patch(
            "torch.load",
            return_value=dict(config=dict(model=dict(backbone=self.inference_task.config.model.backbone)), model={}),
        )

        self.inference_task.load_model(mocker_otx_model)

        mocker_segment_anything.assert_called_once()
        mocker_io_bytes_io.assert_called_once()
        mocker_torch_load.assert_called_once()

    @e2e_pytest_unit
    def test_infer(self, mocker):
        """Test infer."""
        mocker_trainer = mocker.patch("otx.algorithms.visual_prompting.tasks.inference.Trainer")

        dataset = generate_visual_prompting_dataset()
        model = ModelEntity(dataset, self.task_environment.get_model_configuration())

        self.inference_task.infer(dataset, model)

        mocker_trainer.assert_called_once()

    @e2e_pytest_unit
    def test_evaluate(self, mocker):
        """Test evaluate."""
        mocker_dice_average = mocker.patch("otx.api.usecases.evaluation.metrics_helper.DiceAverage")
        validation_dataset = generate_visual_prompting_dataset()

        resultset = ResultSetEntity(
            model=self.task_environment.model,
            ground_truth_dataset=validation_dataset,
            prediction_dataset=validation_dataset,
        )

        self.inference_task.evaluate(resultset)

        mocker_dice_average.assert_called_once()
        assert not isinstance(resultset.performance, NullPerformance)

    @e2e_pytest_unit
    def test_model_info(self, mocker, monkeypatch):
        """Test model_info."""
        setattr(self.inference_task, "trainer", None)
        mocker.patch.object(self.inference_task, "trainer")

        model_info = self.inference_task.model_info()

        assert "model" in model_info
        assert isinstance(model_info["model"], OrderedDict)
        assert "config" in model_info
        assert isinstance(model_info["config"], DictConfig)
        assert "version" in model_info

    @e2e_pytest_unit
    def test_save_model(self, mocker):
        """Test save_model."""
        mocker.patch.object(self.inference_task, "model_info")
        mocker_otx_model = mocker.patch("otx.api.entities.model.ModelEntity")
        mocker_io_bytes_io = mocker.patch("io.BytesIO")
        mocker_torch_save = mocker.patch("torch.save")

        self.inference_task.save_model(mocker_otx_model)

        mocker_io_bytes_io.assert_called_once()
        mocker_torch_save.assert_called_once()
