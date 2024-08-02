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
from otx.algo.detection.atss import MobileNetV2ATSS
from otx.algo.explain.explain_algo import feature_vector_fn
from otx.core.metrics.fmeasure import FMeasureCallable
from otx.core.types.export import TaskLevelExportParameters
from torch.optim import Optimizer

if TYPE_CHECKING:
    from omegaconf.dictconfig import DictConfig


class TestOTXDetectionModel:
    @pytest.fixture()
    def mock_optimizer(self):
        return lambda _: create_autospec(Optimizer)

    @pytest.fixture()
    def mock_scheduler(self):
        return lambda _: create_autospec([ReduceLROnPlateau])

    @pytest.fixture(
        params=[
            {
                "confidence_threshold": 0.35,
                "state_dict": {},
            },
            {
                "hyper_parameters": {"best_confidence_threshold": 0.35},
                "state_dict": {},
            },
        ],
        ids=["v1", "v2"],
    )
    def mock_ckpt(self, request):
        return request.param

    @pytest.fixture()
    def config(self) -> DictConfig:
        cfg_path = files("otx") / "algo" / "detection" / "mmconfigs" / "yolox_tiny.yaml"
        return OmegaConf.load(cfg_path)

    @pytest.fixture()
    def otx_model(self) -> MobileNetV2ATSS:
        return MobileNetV2ATSS(label_info=1)

    def test_configure_metric_with_ckpt(
        self,
        mock_optimizer,
        mock_scheduler,
        mock_ckpt,
    ) -> None:
        model = MobileNetV2ATSS(
            label_info=1,
            torch_compile=False,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
            metric=FMeasureCallable,
        )

        model.on_load_checkpoint(mock_ckpt)

        assert model.hparams["best_confidence_threshold"] == 0.35

    def test_create_model(self, otx_model) -> None:
        mmdet_model = otx_model._create_model()
        assert mmdet_model is not None
        assert isinstance(mmdet_model, torch.nn.Module)

    def test_get_num_anchors(self, otx_model):
        num_anchors = otx_model.get_num_anchors()
        assert isinstance(num_anchors, list)
        assert all(isinstance(n, int) for n in num_anchors)

    def test_get_explain_fn(self, otx_model):
        otx_model.explain_mode = True
        explain_fn = otx_model.get_explain_fn()
        assert callable(explain_fn)

    def test_forward_explain_detection(self, otx_model, fxt_det_data_entity):
        otx_model.model.feature_vector_fn = feature_vector_fn
        otx_model.model.explain_fn = otx_model.get_explain_fn()
        inputs = fxt_det_data_entity[2]
        inputs.images = torch.randn(1, 3, 64, 64)
        result = otx_model._forward_explain_detection(otx_model.model, inputs, mode="predict")

        assert "predictions" in result
        assert "feature_vector" in result
        assert "saliency_map" in result

    def test_customize_inputs(self, otx_model, fxt_det_data_entity) -> None:
        output_data = otx_model._customize_inputs(fxt_det_data_entity[2])
        assert output_data["mode"] == "loss"
        assert output_data["entity"] == fxt_det_data_entity[2]

    def test_forward_explain(self, otx_model, fxt_det_data_entity):
        inputs = fxt_det_data_entity[2]
        otx_model.training = False
        otx_model.explain_mode = True
        outputs = otx_model.forward_explain(inputs)

        assert outputs.has_xai_outputs
        assert outputs.feature_vector is not None
        assert outputs.saliency_map is not None

    def test_reset_restore_model_forward(self, otx_model):
        otx_model.explain_mode = True
        initial_model_forward = otx_model.model.forward

        otx_model._reset_model_forward()
        assert otx_model.original_model_forward is not None
        assert str(otx_model.model.forward) != str(otx_model.original_model_forward)

        otx_model._restore_model_forward()
        assert otx_model.original_model_forward is None
        assert str(otx_model.model.forward) == str(initial_model_forward)

    def test_export_parameters(self, otx_model):
        parameters = otx_model._export_parameters
        assert isinstance(parameters, TaskLevelExportParameters)
        assert parameters.task_type == "detection"

    def test_dummy_input(self, otx_model: MobileNetV2ATSS):
        batch_size = 2
        batch = otx_model.get_dummy_input(batch_size)
        assert batch.batch_size == batch_size
