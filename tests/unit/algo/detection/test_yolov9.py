# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of YOLOv9 architecture."""

from unittest.mock import Mock, patch

from otx.algo.detection.backbones.gelan import GELANModule
from otx.algo.detection.heads.yolo_head import YOLOv9HeadModule
from otx.algo.detection.necks.yolo_neck import YOLOv9NeckModule
from otx.algo.detection.yolov9 import YOLOv9
from otx.core.exporter.native import OTXNativeModelExporter


class TestYOLOv9:
    def test_init(self) -> None:
        otx_yolov9_s = YOLOv9(model_name="yolov9_s", label_info=3)
        assert isinstance(otx_yolov9_s.model.backbone, GELANModule)
        assert isinstance(otx_yolov9_s.model.neck, YOLOv9NeckModule)
        assert isinstance(otx_yolov9_s.model.bbox_head, YOLOv9HeadModule)
        assert otx_yolov9_s.input_size == (640, 640)

        otx_yolov9_m = YOLOv9(model_name="yolov9_m", label_info=3)
        assert otx_yolov9_m.input_size == (640, 640)

        otx_yolov9_m = YOLOv9(model_name="yolov9_m", label_info=3, input_size=(416, 416))
        assert otx_yolov9_m.input_size == (416, 416)

    def test_exporter(self) -> None:
        otx_yolov9_s = YOLOv9(model_name="yolov9_s", label_info=3)
        otx_yolov9_s_exporter = otx_yolov9_s._exporter
        assert isinstance(otx_yolov9_s_exporter, OTXNativeModelExporter)
        assert otx_yolov9_s_exporter.swap_rgb is True

    def test_to(self) -> None:
        model = YOLOv9(model_name="yolov9_s", label_info=3)
        model.vec2box.update = Mock()
        model.model.criterion.vec2box.update = Mock()
        model.model.criterion.matcher.anchors.to = Mock()
        model.model.criterion.loss_dfl.anchors_norm.to = Mock()

        with patch("torch.nn.Module.to") as mock_to:
            ret = model.to("cuda")

            assert mock_to.called
            assert mock_to.call_args == (("cuda",), {})
            assert ret.vec2box.update.called
            assert ret.model.criterion.vec2box.update.called
