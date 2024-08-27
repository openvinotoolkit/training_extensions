# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of OTX YOLOX architecture."""

import pytest
import torch
from otx.algo.detection.backbones.csp_darknet import CSPDarknetModule
from otx.algo.detection.heads.yolox_head import YOLOXHeadModule
from otx.algo.detection.necks.yolox_pafpn import YOLOXPAFPNModule
from otx.algo.detection.yolox import YOLOX
from otx.core.data.entity.detection import DetBatchPredEntity
from otx.core.exporter.native import OTXNativeModelExporter


class TestYOLOX:
    def test_init(self) -> None:
        otx_yolox_l = YOLOX(model_name="yolox_l", label_info=3)
        assert isinstance(otx_yolox_l.model.backbone, CSPDarknetModule)
        assert isinstance(otx_yolox_l.model.neck, YOLOXPAFPNModule)
        assert isinstance(otx_yolox_l.model.bbox_head, YOLOXHeadModule)
        assert otx_yolox_l.input_size == (640, 640)

        otx_yolox_tiny = YOLOX(model_name="yolox_tiny", label_info=3)
        assert otx_yolox_tiny.input_size == (640, 640)

        otx_yolox_tiny = YOLOX(model_name="yolox_tiny", label_info=3, input_size=(416, 416))
        assert otx_yolox_tiny.input_size == (416, 416)

    def test_exporter(self) -> None:
        otx_yolox_l = YOLOX(model_name="yolox_l", label_info=3)
        otx_yolox_l_exporter = otx_yolox_l._exporter
        assert isinstance(otx_yolox_l_exporter, OTXNativeModelExporter)
        assert otx_yolox_l_exporter.swap_rgb is True

        otx_yolox_tiny = YOLOX(model_name="yolox_tiny", label_info=3)
        otx_yolox_tiny_exporter = otx_yolox_tiny._exporter
        assert isinstance(otx_yolox_tiny_exporter, OTXNativeModelExporter)
        assert otx_yolox_tiny_exporter.swap_rgb is False

    @pytest.mark.parametrize(
        "model",
        [
            YOLOX(model_name="yolox_tiny", label_info=3),
            YOLOX(model_name="yolox_s", label_info=3),
            YOLOX(model_name="yolox_l", label_info=3),
            YOLOX(model_name="yolox_x", label_info=3),
        ],
    )
    def test_loss(self, model, fxt_data_module):
        data = next(iter(fxt_data_module.train_dataloader()))
        data.images = [torch.randn(3, 32, 32), torch.randn(3, 48, 48)]
        output = model(data)
        assert "loss_cls" in output
        assert "loss_bbox" in output
        assert "loss_obj" in output

    @pytest.mark.parametrize(
        "model",
        [
            YOLOX(model_name="yolox_tiny", label_info=3),
            YOLOX(model_name="yolox_s", label_info=3),
            YOLOX(model_name="yolox_l", label_info=3),
            YOLOX(model_name="yolox_x", label_info=3),
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
            YOLOX(model_name="yolox_tiny", label_info=3),
            YOLOX(model_name="yolox_s", label_info=3),
            YOLOX(model_name="yolox_l", label_info=3),
            YOLOX(model_name="yolox_x", label_info=3),
        ],
    )
    def test_export(self, model):
        model.eval()
        output = model.forward_for_tracing(torch.randn(1, 3, 32, 32))
        assert len(output) == 2

        model.explain_mode = True
        output = model.forward_for_tracing(torch.randn(1, 3, 32, 32))
        assert len(output) == 4
