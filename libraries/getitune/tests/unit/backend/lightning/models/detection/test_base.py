# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for detection model module."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import create_autospec

import pytest
import torch
from importlib_resources import files
from lightning.pytorch.cli import ReduceLROnPlateau
from omegaconf import OmegaConf
from torch.optim import Optimizer

from getitune.backend.lightning.models.base import DataInputParams
from getitune.backend.lightning.models.detection.atss import ATSS
from getitune.backend.lightning.tools.explain.explain_algo import feature_vector_fn
from getitune.metrics.fmeasure import FMeasureCallable
from getitune.types.export import TaskLevelExportParameters

if TYPE_CHECKING:
    from omegaconf.dictconfig import DictConfig


class TestLightningDetectionModel:
    @pytest.fixture
    def mock_optimizer(self):
        return lambda _: create_autospec(Optimizer)

    @pytest.fixture
    def mock_scheduler(self):
        return lambda _: create_autospec([ReduceLROnPlateau])

    @pytest.fixture
    def mock_ckpt(self, request):
        return {
            "hyper_parameters": {"best_confidence_threshold": 0.35},
            "state_dict": {},
        }

    @pytest.fixture
    def config(self) -> DictConfig:
        cfg_path = files("getitune") / "algo" / "detection" / "mmconfigs" / "yolox_tiny.yaml"
        return OmegaConf.load(cfg_path)

    @pytest.fixture
    def model(self) -> ATSS:
        return ATSS(
            model_name="atss_mobilenetv2",
            label_info=1,
            data_input_params=DataInputParams((320, 320), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )

    def test_configure_metric_with_ckpt(
        self,
        mock_optimizer,
        mock_scheduler,
        mock_ckpt,
    ) -> None:
        model = ATSS(
            model_name="atss_mobilenetv2",
            label_info=2,
            data_input_params=DataInputParams((320, 320), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            torch_compile=False,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
            metric=FMeasureCallable,
        )

        model.on_load_checkpoint(mock_ckpt)

        assert model.hparams["best_confidence_threshold"] == 0.35

    def test_create_model(self, model) -> None:
        mmdet_model = model._create_model()
        assert mmdet_model is not None
        assert isinstance(mmdet_model, torch.nn.Module)

    def test_get_num_anchors(self, model):
        num_anchors = model.get_num_anchors()
        assert isinstance(num_anchors, list)
        assert all(isinstance(n, int) for n in num_anchors)

    def test_get_explain_fn(self, model):
        model.explain_mode = True
        explain_fn = model.get_explain_fn()
        assert callable(explain_fn)

    def test_forward_explain_detection(self, model, fxt_det_data_entity):
        model.model.feature_vector_fn = feature_vector_fn
        model.model.explain_fn = model.get_explain_fn()
        inputs = fxt_det_data_entity[2]
        inputs.images = torch.randn(1, 3, 64, 64)
        result = model._forward_explain_detection(model.model, inputs, mode="predict")

        assert "predictions" in result
        assert "feature_vector" in result
        assert "saliency_map" in result

    def test_customize_inputs(self, model, fxt_det_data_entity) -> None:
        output_data = model._customize_inputs(fxt_det_data_entity[2])
        assert output_data["mode"] == "loss"
        assert output_data["entity"] == fxt_det_data_entity[2]

    def test_forward_explain(self, model, fxt_det_data_entity):
        inputs = fxt_det_data_entity[2]
        model.training = False
        model.explain_mode = True
        outputs = model.forward_explain(inputs)

        assert outputs.saliency_map is not None
        assert len(outputs.saliency_map) > 0
        assert outputs.feature_vector is not None
        assert outputs.saliency_map is not None

    def test_reset_restore_model_forward(self, model):
        model.explain_mode = True
        initial_model_forward = model.model.forward

        model._reset_model_forward()
        assert model.original_model_forward is not None
        assert str(model.model.forward) != str(model.original_model_forward)

        model._restore_model_forward()
        assert model.original_model_forward is None
        assert str(model.model.forward) == str(initial_model_forward)

    def test_export_parameters(self, model):
        parameters = model._export_parameters
        assert isinstance(parameters, TaskLevelExportParameters)
        assert parameters.task_type == "detection"

    def test_export_parameters_without_nms(self, model):
        model.export_nms = False
        parameters = model._export_parameters
        assert isinstance(parameters, TaskLevelExportParameters)
        assert parameters.nms_execute is True
        # agnostic_nms and nms_max_predictions are not set; ModelAPI defaults suffice
        assert parameters.agnostic_nms is None
        assert parameters.nms_max_predictions is None
        metadata = parameters.to_metadata()
        assert metadata[("model_info", "nms_execute")] == "True"
        assert ("model_info", "agnostic_nms") not in metadata
        assert ("model_info", "nms_max_predictions") not in metadata

    def test_export_nms_defaults_to_false(self, model):
        assert model.export_nms is False

    def test_export_nms_can_be_enabled(self):
        model = ATSS(
            model_name="atss_mobilenetv2",
            label_info=1,
            data_input_params=DataInputParams((320, 320), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            export_nms=True,
        )
        assert model.export_nms is True

    def test_export_parameters_with_nms(self, model):
        model.export_nms = True
        parameters = model._export_parameters
        assert parameters.nms_execute is False
        assert parameters.agnostic_nms is None
        assert parameters.nms_max_predictions is None

    def test_forward_for_tracing_without_nms(self, model):
        model.eval()
        model.export_nms = False
        output = model.forward_for_tracing(torch.randn(1, 3, 64, 64))
        assert len(output) == 2
        dets, labels = output
        # Without NMS: dets=(1, num_priors, 5), labels=(1, num_priors)
        assert dets.ndim == 3
        assert dets.shape[0] == 1
        assert dets.shape[2] == 5
        assert labels.ndim == 2
        assert labels.shape == dets.shape[:2]

    def test_forward_for_tracing_with_nms(self, model):
        model.eval()
        model.export_nms = True

        dets, labels = model.forward_for_tracing(torch.randn(1, 3, 64, 64))

        assert dets.ndim == 3
        assert dets.shape[0] == 1
        assert dets.shape[2] == 5
        assert labels.ndim == 2
        assert labels.shape == dets.shape[:2]
        assert dets.shape[1] <= model.model.test_cfg["max_per_img"]

        scores = dets[..., 4]
        assert torch.all((scores >= 0) & (scores <= 1))
        assert torch.all(scores[:, :-1] >= scores[:, 1:])
        assert torch.all((labels >= 0) & (labels < model.num_classes))

    def test_dummy_input(self, model: ATSS):
        batch_size = 2
        batch = model.get_dummy_input(batch_size)
        assert batch.batch_size == batch_size
