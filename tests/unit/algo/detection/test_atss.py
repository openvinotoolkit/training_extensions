# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of OTX SSD architecture."""

import pytest
import torch
from otx.algo.detection.atss import ATSS
from otx.algo.utils.support_otx_v1 import OTXv1Helper
from otx.core.data.entity.detection import DetBatchPredEntity
from otx.core.exporter.native import OTXModelExporter
from otx.core.types.export import TaskLevelExportParameters
from torch._dynamo.testing import CompileCounter


class TestATSS:
    def test(self, mocker) -> None:
        model = ATSS(model_name="atss_mobilenetv2", label_info=2)
        mock_load_ckpt = mocker.patch.object(OTXv1Helper, "load_det_ckpt")
        model.load_from_otx_v1_ckpt({})
        mock_load_ckpt.assert_called_once_with({}, "model.")

        assert isinstance(model._export_parameters, TaskLevelExportParameters)
        assert isinstance(model._exporter, OTXModelExporter)

    @pytest.mark.parametrize(
        "model",
        [
            ATSS(model_name="atss_mobilenetv2", label_info=3),
            ATSS(model_name="atss_resnext101", label_info=3),
        ],
    )
    def test_loss(self, model, fxt_data_module):
        data = next(iter(fxt_data_module.train_dataloader()))
        data.images = [torch.randn(3, 32, 32), torch.randn(3, 48, 48)]
        output = model(data)
        assert "loss_cls" in output
        assert "loss_bbox" in output
        assert "loss_centerness" in output

    @pytest.mark.parametrize(
        "model",
        [
            ATSS(model_name="atss_mobilenetv2", label_info=3),
            ATSS(model_name="atss_resnext101", label_info=3),
        ],
    )
    def test_predict(self, model, fxt_data_module):
        data = next(iter(fxt_data_module.train_dataloader()))
        data.images = [torch.randn(3, 32, 32), torch.randn(3, 48, 48)]
        model.eval()
        output = model(data)
        assert isinstance(output, DetBatchPredEntity)

    @pytest.mark.parametrize(
        "model",
        [
            ATSS(model_name="atss_mobilenetv2", label_info=3),
            ATSS(model_name="atss_resnext101", label_info=3),
        ],
    )
    def test_export(self, model):
        model.eval()
        output = model.forward_for_tracing(torch.randn(1, 3, 32, 32))
        assert len(output) == 2

        model.explain_mode = True
        output = model.forward_for_tracing(torch.randn(1, 3, 32, 32))
        assert len(output) == 4

    @pytest.mark.parametrize(
        "model",
        [
            ATSS(model_name="atss_mobilenetv2", label_info=3),
            ATSS(model_name="atss_resnext101", label_info=3),
        ],
    )
    def test_compiled_model(self, model):
        # Set Compile Counter
        torch._dynamo.reset()
        cnt = CompileCounter()

        # Set model compile setting
        model.model = torch.compile(model.model, backend=cnt)

        # Prepare inputs
        x = torch.randn(1, 3, *model.input_size)
        model.model(x)
        assert cnt.frame_count == 6
