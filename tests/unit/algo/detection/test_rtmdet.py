# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of RTMDet."""

import pytest
import torch
from otx.algo.common.backbones.cspnext import CSPNeXtModule
from otx.algo.detection.heads.rtmdet_head import RTMDetSepBNHeadModule
from otx.algo.detection.necks.cspnext_pafpn import CSPNeXtPAFPNModule
from otx.algo.detection.rtmdet import RTMDet
from otx.core.data.entity.detection import DetBatchPredEntity
from otx.core.exporter.native import OTXNativeModelExporter


class TestRTMDet:
    def test_init(self) -> None:
        otx_rtmdet_tiny = RTMDet(model_name="rtmdet_tiny", label_info=3)
        assert isinstance(otx_rtmdet_tiny.model.backbone, CSPNeXtModule)
        assert isinstance(otx_rtmdet_tiny.model.neck, CSPNeXtPAFPNModule)
        assert isinstance(otx_rtmdet_tiny.model.bbox_head, RTMDetSepBNHeadModule)
        assert otx_rtmdet_tiny.input_size == (640, 640)

    def test_exporter(self) -> None:
        otx_rtmdet_tiny = RTMDet(model_name="rtmdet_tiny", label_info=3)
        otx_rtmdet_tiny_exporter = otx_rtmdet_tiny._exporter
        assert isinstance(otx_rtmdet_tiny_exporter, OTXNativeModelExporter)
        assert otx_rtmdet_tiny_exporter.swap_rgb is True

    @pytest.mark.parametrize("model", [RTMDet(model_name="rtmdet_tiny", label_info=3)])
    def test_loss(self, model, fxt_data_module):
        data = next(iter(fxt_data_module.train_dataloader()))
        data.images = [torch.randn(3, 32, 32), torch.randn(3, 48, 48)]
        output = model(data)
        assert "loss_cls" in output
        assert "loss_bbox" in output

    @pytest.mark.parametrize("model", [RTMDet(model_name="rtmdet_tiny", label_info=3)])
    def test_predict(self, model, fxt_data_module):
        data = next(iter(fxt_data_module.train_dataloader()))
        data.images = [torch.randn(3, 32, 32), torch.randn(3, 48, 48)]
        model.eval()
        output = model(data)
        assert isinstance(output, DetBatchPredEntity)

    @pytest.mark.parametrize("model", [RTMDet(model_name="rtmdet_tiny", label_info=3)])
    def test_export(self, model):
        model.eval()
        output = model.forward_for_tracing(torch.randn(1, 3, 32, 32))
        assert len(output) == 2

        model.explain_mode = True
        output = model.forward_for_tracing(torch.randn(1, 3, 32, 32))
        assert len(output) == 4
