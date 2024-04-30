# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of OTX YOLOX architecture."""

from otx.algo.detection.backbones.csp_darknet import CSPDarknet
from otx.algo.detection.heads.yolox_head import YOLOXHead
from otx.algo.detection.necks.yolox_pafpn import YOLOXPAFPN
from otx.algo.detection.yolox import YOLOXL, YOLOXTINY
from otx.core.exporter.native import OTXNativeModelExporter


class TestYOLOX:
    def test_init(self) -> None:
        otx_yolox_l = YOLOXL(label_info=3)
        assert isinstance(otx_yolox_l.model.backbone, CSPDarknet)
        assert isinstance(otx_yolox_l.model.neck, YOLOXPAFPN)
        assert isinstance(otx_yolox_l.model.bbox_head, YOLOXHead)
        assert otx_yolox_l.image_size == (1, 3, 640, 640)
        assert otx_yolox_l.tile_image_size == (1, 3, 640, 640)

        otx_yolox_tiny = YOLOXTINY(label_info=3)
        assert otx_yolox_tiny.image_size == (1, 3, 416, 416)
        assert otx_yolox_tiny.tile_image_size == (1, 3, 416, 416)

    def test_exporter(self) -> None:
        otx_yolox_l = YOLOXL(label_info=3)
        otx_yolox_l_exporter = otx_yolox_l._exporter
        assert isinstance(otx_yolox_l_exporter, OTXNativeModelExporter)
        assert otx_yolox_l_exporter.swap_rgb is True

        otx_yolox_tiny = YOLOXTINY(label_info=3)
        otx_yolox_tiny_exporter = otx_yolox_tiny._exporter
        assert isinstance(otx_yolox_tiny_exporter, OTXNativeModelExporter)
        assert otx_yolox_tiny_exporter.swap_rgb is False
