# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of YOLOv9 architecture."""

import re
from unittest.mock import Mock, patch

import torch
from otx.algo.detection.backbones.gelan import GELANModule
from otx.algo.detection.heads.yolo_head import YOLOHeadModule
from otx.algo.detection.necks.yolo_neck import YOLONeckModule
from otx.algo.detection.yolov9 import YOLOv9, _load_from_state_dict_for_yolov9
from otx.core.exporter.native import OTXNativeModelExporter


def test_load_from_state_dict_for_yolov9() -> None:
    model = YOLOv9(model_name="yolov9_s", label_info=3)
    model._load_from_state_dict = Mock()
    state_dict = {
        "0.conv.weight": torch.randn(32, 3, 3, 3),
        "0.bn.weight": torch.randn(32),
        "0.bn.bias": torch.randn(32),
        "0.bn.running_mean": torch.randn(32),
        "0.bn.running_var": torch.randn(32),
        "0.bn.num_batches_tracked": torch.randn([]),
        "9.conv1.conv.weight": torch.randn(128, 256, 1, 1),
        "9.conv1.bn.weight": torch.randn(128),
        "9.conv1.bn.bias": torch.randn(128),
        "9.conv1.bn.running_mean": torch.randn(128),
        "9.conv1.bn.running_var": torch.randn(128),
        "9.conv1.bn.num_batches_tracked": torch.randn([]),
        "15.conv1.conv.weight": torch.randn(128, 320, 1, 1),
        "15.conv1.bn.weight": torch.randn(128),
        "15.conv1.bn.bias": torch.randn(128),
        "15.conv1.bn.running_mean": torch.randn(128),
        "15.conv1.bn.running_var": torch.randn(128),
        "15.conv1.bn.num_batches_tracked": torch.randn([]),
    }
    updated_state_dict = state_dict.copy()
    prefix = ""
    local_metadata = {}
    strict = True
    missing_keys = []
    unexpected_keys = []
    error_msgs = []

    _load_from_state_dict_for_yolov9(
        model.model,
        updated_state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    )

    backbone_len: int = len(model.model.backbone.module)
    neck_len: int = len(model.model.neck.module)  # type: ignore[union-attr]
    for (k, v), (updated_k, updated_v) in zip(state_dict.items(), updated_state_dict.items()):
        match = re.match(r"^(\d+)\.(.*)$", k)
        orig_idx = int(match.group(1))
        if orig_idx < backbone_len:
            assert re.match(r"backbone.module.", updated_k)
        elif orig_idx < backbone_len + neck_len:
            assert re.match(r"neck.module.", updated_k)
        else:  # for bbox_head
            assert re.match(r"bbox_head.module.", updated_k)

        assert torch.allclose(v, updated_v)
        assert v.shape == updated_v.shape


class TestYOLOv9:
    def test_init(self) -> None:
        otx_yolov9_s = YOLOv9(model_name="yolov9_s", label_info=3)
        assert isinstance(otx_yolov9_s.model.backbone, GELANModule)
        assert isinstance(otx_yolov9_s.model.neck, YOLONeckModule)
        assert isinstance(otx_yolov9_s.model.bbox_head, YOLOHeadModule)
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
