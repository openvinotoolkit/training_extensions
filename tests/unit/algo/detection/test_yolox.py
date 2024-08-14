# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of OTX YOLOX architecture."""

import pytest
import torch
from otx.algo.detection.backbones import CSPDarknet
from otx.algo.detection.heads import YOLOXHead
from otx.algo.detection.necks import YOLOXPAFPN
from otx.algo.detection.yolox import YOLOXL, YOLOXS, YOLOXTINY, YOLOXX
from otx.core.data.entity.detection import DetBatchPredEntity
from otx.core.exporter.native import OTXNativeModelExporter


class TestYOLOX:
    def test_init(self) -> None:
        otx_yolox_l = YOLOXL(label_info=3)
        assert isinstance(otx_yolox_l.model.backbone, CSPDarknet)
        assert isinstance(otx_yolox_l.model.neck, YOLOXPAFPN)
        assert isinstance(otx_yolox_l.model.bbox_head, YOLOXHead)
        assert otx_yolox_l.input_size == (640, 640)

        otx_yolox_tiny = YOLOXTINY(label_info=3)
        assert otx_yolox_tiny.input_size == (416, 416)

    def test_exporter(self) -> None:
        otx_yolox_l = YOLOXL(label_info=3)
        otx_yolox_l_exporter = otx_yolox_l._exporter
        assert isinstance(otx_yolox_l_exporter, OTXNativeModelExporter)
        assert otx_yolox_l_exporter.swap_rgb is True

        otx_yolox_tiny = YOLOXTINY(label_info=3)
        otx_yolox_tiny_exporter = otx_yolox_tiny._exporter
        assert isinstance(otx_yolox_tiny_exporter, OTXNativeModelExporter)
        assert otx_yolox_tiny_exporter.swap_rgb is False

    @pytest.mark.parametrize("model", [YOLOXTINY(3), YOLOXS(3), YOLOXL(3), YOLOXX(3)])
    def test_loss(self, model, fxt_data_module):
        data = next(iter(fxt_data_module.train_dataloader()))
        data.images = [torch.randn(3, 32, 32), torch.randn(3, 48, 48)]
        output = model(data)
        assert "loss_cls" in output
        assert "loss_bbox" in output
        assert "loss_obj" in output

    @pytest.mark.parametrize("model", [YOLOXTINY(3), YOLOXS(3), YOLOXL(3), YOLOXX(3)])
    def test_predict(self, model, fxt_data_module):
        data = next(iter(fxt_data_module.train_dataloader()))
        data.images = [torch.randn(3, 32, 32), torch.randn(3, 48, 48)]
        model.eval()
        output = model(data)
        assert isinstance(output, DetBatchPredEntity)

    @pytest.mark.parametrize("model", [YOLOXTINY(3), YOLOXS(3), YOLOXL(3), YOLOXX(3)])
    def test_export(self, model):
        model.eval()
        output = model.forward_for_tracing(torch.randn(1, 3, 32, 32))
        assert len(output) == 2

        model.explain_mode = True
        output = model.forward_for_tracing(torch.randn(1, 3, 32, 32))
        assert len(output) == 4
