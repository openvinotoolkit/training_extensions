"""Tests the methods in the inference task."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from collections import OrderedDict
from typing import Optional

import pytest
import logging
from omegaconf import DictConfig

from otx.algorithms.visual_prompting.tasks.inference import InferenceTask
from otx.api.entities.metrics import NullPerformance
from otx.api.entities.model import ModelEntity
from otx.api.entities.resultset import ResultSetEntity
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from otx.algorithms.common.utils.logger import get_logger
from tests.unit.algorithms.visual_prompting.test_helpers import (
    generate_visual_prompting_dataset,
    init_environment,
)

logger = get_logger()


class TestInferenceTask:
    @pytest.fixture
    def load_inference_task(self, tmpdir, mocker):
        def _load_inference_task(path: Optional[str] = None, resume: bool = False):
            if path is None:
                mocker_model = None
            else:
                mocker_model = mocker.patch("otx.api.entities.model.ModelEntity")
                mocker_model.model_adapters = {}
                mocker.patch.dict(mocker_model.model_adapters, {"path": path, "resume": resume})

            self.task_environment = init_environment(mocker_model)
            output_path = str(tmpdir.mkdir("visual_prompting_training_test"))

            return InferenceTask(self.task_environment, output_path)

        return _load_inference_task

    @e2e_pytest_unit
    @pytest.mark.parametrize("resume", [False, True])
    @pytest.mark.parametrize("path", [None, "checkpoint.ckpt", "checkpoint.pth"])
    def test_get_config(self, mocker, load_inference_task, path: Optional[str], resume: bool):
        """Test get_config."""
        mocker.patch("otx.algorithms.visual_prompting.tasks.inference.InferenceTask.load_model")
        inference_task = load_inference_task(path, resume)

        assert isinstance(inference_task.config, DictConfig)
        assert inference_task.config.dataset.task == "visual_prompting"
        if path:
            if path.endswith(".pth"):
                # TODO (sungchul): when applying resume
                # pytorch weights
                assert inference_task.config.model.checkpoint == path
                assert inference_task.config.trainer.resume_from_checkpoint is None
            elif path.endswith(".ckpt") and resume:
                # resume with pytorch lightning weights
                assert inference_task.config.model.checkpoint != path  # use default checkpoint
                assert inference_task.config.trainer.resume_from_checkpoint == path
            else:
                # just train with pytorch lightning weights
                assert inference_task.config.model.checkpoint == path
                assert inference_task.config.trainer.resume_from_checkpoint is None

    @e2e_pytest_unit
    @pytest.mark.parametrize("path", [None, "checkpoint.ckpt"])
    @pytest.mark.parametrize("resume", [True, False])
    def test_load_model_without_otx_model_or_with_lightning_ckpt(
        self, mocker, load_inference_task, path: str, resume: bool
    ):
        """Test load_model to resume."""
        mocker_segment_anything = mocker.patch(
            "otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.SegmentAnything"
        )

        load_inference_task(path=path, resume=resume)

        mocker_segment_anything.assert_called_once()

    @e2e_pytest_unit
    @pytest.mark.parametrize("resume", [True, False])
    def test_load_model_with_pytorch_pth(self, mocker, load_inference_task, resume: bool):
        """Test load_model with otx_model."""
        mocker_segment_anything = mocker.patch(
            "otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.SegmentAnything"
        )
        mocker_io_bytes_io = mocker.patch("io.BytesIO")
        mocker_torch_load = mocker.patch(
            "torch.load",
            return_value=dict(config=dict(model=dict(backbone="sam_vit_b")), model={}),
        )

        load_inference_task(path="checkpoint.pth", resume=resume)

        mocker_segment_anything.assert_called_once()
        mocker_io_bytes_io.assert_called_once()
        mocker_torch_load.assert_called_once()

    @e2e_pytest_unit
    def test_infer(self, mocker, load_inference_task):
        """Test infer."""
        mocker_trainer = mocker.patch("otx.algorithms.visual_prompting.tasks.inference.Trainer")

        inference_task = load_inference_task()
        dataset = generate_visual_prompting_dataset()
        model = ModelEntity(dataset, inference_task.task_environment.get_model_configuration())

        inference_task.infer(dataset, model)

        mocker_trainer.assert_called_once()

    @e2e_pytest_unit
    def test_evaluate(self, mocker, load_inference_task):
        """Test evaluate."""
        mocker_dice_average = mocker.patch("otx.api.usecases.evaluation.metrics_helper.DiceAverage")

        inference_task = load_inference_task()
        validation_dataset = generate_visual_prompting_dataset()
        resultset = ResultSetEntity(
            model=inference_task.task_environment.model,
            ground_truth_dataset=validation_dataset,
            prediction_dataset=validation_dataset,
        )

        inference_task.evaluate(resultset)

        mocker_dice_average.assert_called_once()
        assert not isinstance(resultset.performance, NullPerformance)

    @e2e_pytest_unit
    def test_model_info(self, mocker, load_inference_task):
        """Test model_info."""
        inference_task = load_inference_task()
        setattr(inference_task, "trainer", None)
        mocker.patch.object(inference_task, "trainer")

        model_info = inference_task.model_info()

        assert "model" in model_info
        assert isinstance(model_info["model"], OrderedDict)
        assert "config" in model_info
        assert isinstance(model_info["config"], DictConfig)
        assert "version" in model_info

    @e2e_pytest_unit
    def test_save_model(self, mocker, load_inference_task):
        """Test save_model."""
        inference_task = load_inference_task()
        mocker.patch.object(inference_task, "model_info")
        mocker_otx_model = mocker.patch("otx.api.entities.model.ModelEntity")
        mocker_io_bytes_io = mocker.patch("io.BytesIO")
        mocker_torch_save = mocker.patch("torch.save")

        inference_task.save_model(mocker_otx_model)

        mocker_io_bytes_io.assert_called_once()
        mocker_torch_save.assert_called_once()
