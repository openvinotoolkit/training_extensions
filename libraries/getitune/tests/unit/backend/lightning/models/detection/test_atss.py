# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of getitune ATSS architecture."""

import pytest
import torch

from getitune.backend.lightning.exporter.native import ModelExporter
from getitune.backend.lightning.models.base import DataInputParams
from getitune.backend.lightning.models.detection.atss import ATSS
from getitune.data.entity.sample import PredictionBatch
from getitune.types.export import TaskLevelExportParameters


class TestATSS:
    @pytest.fixture(params=["atss_mobilenetv2"])
    def fxt_model(self, request) -> ATSS:
        return ATSS(
            model_name=request.param,
            label_info=3,
            data_input_params=DataInputParams((800, 992), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )

    def test(self, mocker) -> None:
        model = ATSS(
            model_name="atss_mobilenetv2",
            label_info=2,
            data_input_params=DataInputParams((800, 992), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )

        assert isinstance(model._export_parameters, TaskLevelExportParameters)
        assert isinstance(model._exporter, ModelExporter)

    def test_loss(self, fxt_model, fxt_detection_batch):
        output = fxt_model(fxt_detection_batch)
        assert "loss_cls" in output
        assert "loss_bbox" in output
        assert "loss_centerness" in output

    def test_predict(self, fxt_model, fxt_detection_batch):
        fxt_model.eval()
        output = fxt_model(fxt_detection_batch)
        assert isinstance(output, PredictionBatch)

    def test_export(self, fxt_model):
        fxt_model.eval()
        output = fxt_model.forward_for_tracing(torch.randn(1, 3, 32, 32))
        assert len(output) == 2

        fxt_model.explain_mode = True
        output = fxt_model.forward_for_tracing(torch.randn(1, 3, 32, 32))
        assert len(output) == 4
