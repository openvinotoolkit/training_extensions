"""Tests the methods in the inference task."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
import torch
import numpy as np
from typing import Optional, Dict, Any

import pytest
from functools import wraps
from omegaconf import DictConfig

from otx.algorithms.visual_prompting.tasks.inference import InferenceTask, ZeroShotTask
from otx.api.usecases.tasks.interfaces.export_interface import ExportType
from otx.api.entities.metrics import NullPerformance
from otx.api.entities.model import ModelEntity, ModelFormat, ModelOptimizationType
from otx.api.entities.resultset import ResultSetEntity
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from otx.algorithms.common.configs.training_base import TrainType
from otx.api.entities.train_parameters import TrainParameters
from otx.utils.logger import get_logger
from tests.unit.algorithms.visual_prompting.test_helpers import (
    generate_visual_prompting_dataset,
    init_environment,
    MockImageEncoder,
)
import onnxruntime

logger = get_logger()


class TestInferenceTask:
    @pytest.fixture
    def load_inference_task(self, tmpdir, mocker):
        def _load_inference_task(
            output_path: Optional[str] = str(tmpdir.mkdir("visual_prompting_inference_test")),
            path: Optional[str] = None,
            resume: bool = False,
            mode: str = "visual_prompt",
        ):
            if path is None:
                mocker_model = None
            else:
                mocker_model = mocker.patch("otx.api.entities.model.ModelEntity")
                mocker_model.model_adapters = {}
                mocker.patch.dict(mocker_model.model_adapters, {"path": path, "resume": resume})

            mocker.patch("pathlib.Path.write_text")
            self.task_environment = init_environment(mocker_model, mode=mode)

            return InferenceTask(self.task_environment, output_path)

        return _load_inference_task

    @e2e_pytest_unit
    @pytest.mark.parametrize("resume", [False, True])
    @pytest.mark.parametrize("path", [None, "checkpoint.ckpt", "checkpoint.pth"])
    def test_get_config_train(self, mocker, load_inference_task, path: Optional[str], resume: bool):
        """Test get_config when train."""
        mocker.patch("otx.algorithms.visual_prompting.tasks.inference.InferenceTask.load_model")
        inference_task = load_inference_task(path=path, resume=resume)

        assert inference_task.mode == "train"
        assert isinstance(inference_task.config, DictConfig)
        assert inference_task.config.dataset.task == "visual_prompting"
        if path:
            if resume:
                assert inference_task.config.trainer.resume_from_checkpoint == path
            else:
                assert inference_task.config.model.checkpoint == path
                assert inference_task.config.trainer.resume_from_checkpoint is None

    @e2e_pytest_unit
    def test_get_config_eval(self, mocker, load_inference_task):
        """Test get_config when eval."""
        mocker.patch("otx.algorithms.visual_prompting.tasks.inference.InferenceTask.load_model")
        inference_task = load_inference_task(output_path=None)

        assert inference_task.mode == "inference"
        assert isinstance(inference_task.config, DictConfig)
        assert inference_task.config.dataset.task == "visual_prompting"

    @e2e_pytest_unit
    @pytest.mark.parametrize("path", [None, "checkpoint.ckpt", "checkpoint.pth"])
    @pytest.mark.parametrize("resume", [True, False])
    @pytest.mark.parametrize(
        "load_return_value",
        [
            {"state_dict": {"layer": "weights"}, "pytorch-lightning_version": "version"},
            {"model": {"layer": "weights"}, "config": {"model": {"backbone": "sam_vit_b"}}},
            {},
        ],
    )
    def test_load_model(self, mocker, load_inference_task, path: str, resume: bool, load_return_value: Dict[str, Any]):
        """Test load_model."""
        mocker_segment_anything = mocker.patch(
            "otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.SegmentAnything"
        )
        mocker_io_bytes_io = mocker.patch("io.BytesIO")
        mocker_torch_load = mocker.patch(
            "torch.load",
            return_value=load_return_value,
        )

        inference_task = load_inference_task(path=path, resume=resume)
        inference_task.load_model(otx_model=inference_task.task_environment.model)

        mocker_segment_anything.assert_called_once()
        if resume or path is None:
            mocker_io_bytes_io.assert_not_called()
            mocker_torch_load.assert_not_called()
        else:
            mocker_io_bytes_io.assert_called_once()
            mocker_torch_load.assert_called_once()

    @e2e_pytest_unit
    def test_load_model_zeroshot(self, mocker, load_inference_task):
        """Test load_model when zero-shot."""
        mocker_segment_anything = mocker.patch(
            "otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.ZeroShotSegmentAnything"
        )

        inference_task = load_inference_task(mode="zero_shot")

        assert inference_task.hyper_parameters.algo_backend.train_type == TrainType.Zeroshot

        model = inference_task.load_model(otx_model=inference_task.task_environment.model)

        mocker_segment_anything.assert_called_once()
        assert "ZeroShotSegmentAnything" in str(model)

    @e2e_pytest_unit
    def test_infer(self, mocker, load_inference_task):
        """Test infer."""
        mocker.patch(
            "otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.visual_prompters.segment_anything.SegmentAnything.load_checkpoint"
        )
        mocker_trainer = mocker.patch("otx.algorithms.visual_prompting.tasks.inference.Trainer")

        inference_task = load_inference_task(output_path=None)
        dataset = generate_visual_prompting_dataset()
        model = ModelEntity(dataset, inference_task.task_environment.get_model_configuration())

        inference_task.infer(dataset, model)

        mocker_trainer.assert_called_once()

    @e2e_pytest_unit
    def test_evaluate(self, mocker, load_inference_task):
        """Test evaluate."""
        mocker.patch(
            "otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.visual_prompters.segment_anything.SegmentAnything.load_checkpoint"
        )
        mocker_dice_average = mocker.patch("otx.api.usecases.evaluation.metrics_helper.DiceAverage")

        inference_task = load_inference_task(output_path=None)
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
        mocker.patch(
            "otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.visual_prompters.segment_anything.SegmentAnything.load_checkpoint"
        )
        mocker.patch("torch.load", return_value={"state_dict": {"layer": "weights"}, "epoch": 0})

        inference_task = load_inference_task(output_path=None)
        inference_task._model_ckpt = "checkpoint"

        model_info = inference_task.model_info()

        assert isinstance(model_info.get("state_dict", None), dict)
        assert model_info.get("state_dict", None)["layer"] == "weights"
        assert isinstance(model_info.get("epoch", None), int)
        assert model_info.get("epoch", None) == 0

    @e2e_pytest_unit
    def test_save_model(self, mocker, load_inference_task):
        """Test save_model."""
        mocker.patch(
            "otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.visual_prompters.segment_anything.SegmentAnything.load_checkpoint"
        )

        inference_task = load_inference_task(output_path=None)
        mocker.patch.object(inference_task, "model_info")
        mocker_otx_model = mocker.patch("otx.api.entities.model.ModelEntity")
        mocker_io_bytes_io = mocker.patch("io.BytesIO")
        mocker_torch_save = mocker.patch("torch.save")

        inference_task.save_model(mocker_otx_model)

        mocker_io_bytes_io.assert_called_once()
        mocker_torch_save.assert_called_once()

    @e2e_pytest_unit
    @pytest.mark.parametrize("export_type", [ExportType.ONNX, ExportType.OPENVINO])
    def test_export(self, mocker, load_inference_task, export_type: ExportType):
        """Test export."""
        mocker.patch(
            "otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.visual_prompters.segment_anything.SegmentAnything.load_checkpoint"
        )

        inference_task = load_inference_task(output_path=None)
        dataset = generate_visual_prompting_dataset()
        output_model = ModelEntity(dataset, inference_task.task_environment.get_model_configuration())

        inference_task.export(export_type, output_model, dump_features=False)

        if export_type == ExportType.ONNX:
            assert output_model.model_format == ModelFormat.ONNX
            output_model.optimization_type = ModelOptimizationType.ONNX
            assert "visual_prompting_image_encoder.onnx" in output_model.model_adapters
            assert "visual_prompting_decoder.onnx" in output_model.model_adapters

        elif export_type == ExportType.OPENVINO:
            assert output_model.model_format == ModelFormat.OPENVINO
            output_model.optimization_type = ModelOptimizationType.ONNX
            assert "visual_prompting_image_encoder.bin" in output_model.model_adapters
            assert "visual_prompting_image_encoder.xml" in output_model.model_adapters
            assert "visual_prompting_decoder.bin" in output_model.model_adapters
            assert "visual_prompting_decoder.xml" in output_model.model_adapters

        assert not output_model.has_xai


class TestZeroShotTask:
    @pytest.fixture(autouse=True)
    def setup(self, tmpdir, mocker):
        mocker.patch("pathlib.Path.write_text")
        self.task_environment = init_environment(mode="zero_shot")

        self.output_path = str(tmpdir.mkdir("visual_prompting_zeroshot_test"))

        self.zero_shot_task = ZeroShotTask(self.task_environment, self.output_path)

    @e2e_pytest_unit
    def test_train(self, mocker):
        """Test train."""
        mocker_trainer = mocker.patch("otx.algorithms.visual_prompting.tasks.inference.Trainer")
        mocker_save = mocker.patch("torch.save")
        mocker.patch.object(self.zero_shot_task, "model_info")

        dataset = generate_visual_prompting_dataset()
        output_model = ModelEntity(
            dataset,
            self.task_environment.get_model_configuration(),
        )

        self.zero_shot_task.train(dataset, output_model, TrainParameters())

        mocker_trainer.assert_called_once()
        mocker_save.assert_called_once()
        assert isinstance(output_model.performance, NullPerformance)
        assert output_model.model_adapters.get("weights.pth", None)
        assert output_model.model_adapters.get("label_schema.json", None)

    @e2e_pytest_unit
    def test_infer(self, mocker):
        """Test infer."""
        mocker.patch(
            "otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.visual_prompters.segment_anything.SegmentAnything.load_checkpoint"
        )
        mocker_trainer = mocker.patch("otx.algorithms.visual_prompting.tasks.inference.Trainer")

        dataset = generate_visual_prompting_dataset()
        model = ModelEntity(dataset, self.zero_shot_task.task_environment.get_model_configuration())

        self.zero_shot_task.infer(dataset, model)

        mocker_trainer.assert_called_once()

    @e2e_pytest_unit
    def test_save_model(self, mocker):
        """Test save_model."""
        mocker.patch(
            "otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.visual_prompters.segment_anything.SegmentAnything.load_checkpoint"
        )

        self.zero_shot_task.model = MockImageEncoder()
        mocker_otx_model = mocker.patch("otx.api.entities.model.ModelEntity")
        mocker_io_bytes_io = mocker.patch("io.BytesIO")
        mocker_torch_save = mocker.patch("torch.save")
        mocker.patch.object(
            self.zero_shot_task.model,
            "state_dict",
            return_value={"reference_info.reference_feats": None, "reference_info.used_indices": None},
        )

        self.zero_shot_task.model.reference_info = "reference_info"

        self.zero_shot_task.save_model(mocker_otx_model)

        mocker_io_bytes_io.assert_called_once()
        mocker_torch_save.assert_called_once()
