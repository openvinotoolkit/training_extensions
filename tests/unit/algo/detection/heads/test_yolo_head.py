# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of YOLOHead architecture."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
import torch
from otx.algo.detection.heads.yolo_head import (
    Anchor2Vec,
    CBFuse,
    CBLinear,
    ImplicitA,
    ImplicitM,
    MultiheadDetection,
    SingleHeadDetectionforYOLOv7,
    SingleHeadDetectionforYOLOv9,
    YOLOHeadModule,
)
from otx.algo.detection.utils.utils import Vec2Box
from otx.algo.utils.mmengine_utils import InstanceData


class TestAnchor2Vec:
    def test_forward(self) -> None:
        anchor_x = torch.randn(2, 64, 5, 5)
        anchor2vec = Anchor2Vec(reg_max=16)
        anchor_x_out, vector_x_out = anchor2vec(anchor_x)

        assert anchor_x_out.shape == (2, 16, 4, 5, 5)
        assert vector_x_out.shape == (2, 4, 5, 5)


class TestCBLinear:
    def test_forward(self) -> None:
        in_channels = 64
        out_channels = [32, 32]
        kernel_size = 3

        cblinear = CBLinear(in_channels, out_channels, kernel_size)

        x = torch.randn(2, in_channels, 10, 10)
        output = cblinear(x)

        assert len(output) == len(out_channels)
        for i, out_channel in enumerate(out_channels):
            assert output[i].shape == (2, out_channel, 10, 10)


class TestCBFuse:
    def test_forward(self) -> None:
        cbfuse = CBFuse(index=[0, 0], mode="nearest")
        x_list = [(torch.randn(2, 64, 10, 10),), (torch.randn(2, 64, 10, 10),), torch.randn(2, 64, 5, 5)]
        output = cbfuse(x_list)

        assert output.shape == (2, 64, 5, 5)


class TestImplicitA:
    def test_forward(self) -> None:
        channel = 64
        mean = 0.0
        std = 0.02

        implicit_a = ImplicitA(channel, mean, std)

        x = torch.randn(2, channel, 10, 10)
        output = implicit_a(x)

        assert output.shape == (2, channel, 10, 10)


class TestImplicitM:
    def test_forward(self) -> None:
        channel = 64
        mean = 1.0
        std = 0.02

        implicit_m = ImplicitM(channel, mean, std)

        x = torch.randn(2, channel, 10, 10)
        output = implicit_m(x)

        assert output.shape == (2, channel, 10, 10)


class TestSingleHeadDetection:
    def test_forward(self):
        in_channels = (64, 128)
        num_classes = 10
        reg_max = 16
        use_group = True

        head = SingleHeadDetectionforYOLOv9(in_channels, num_classes, reg_max=reg_max, use_group=use_group)

        x = torch.randn(2, in_channels[1], 10, 10)
        class_x, anchor_x, vector_x = head(x)

        assert class_x.shape == (2, num_classes, 10, 10)
        assert anchor_x.shape == (2, reg_max, 4, 10, 10)
        assert vector_x.shape == (2, 4, 10, 10)


class TestISingleHeadDetection:
    def test_forward(self) -> None:
        in_channels = 64
        num_classes = 10
        anchor_num = 3

        head = SingleHeadDetectionforYOLOv7(in_channels, num_classes, anchor_num=anchor_num)

        x = torch.randn(2, in_channels, 10, 10)
        output = head(x)

        assert output.shape == (2, anchor_num * (num_classes + 5), 10, 10)


class TestMultiheadDetection:
    def test_forward(self):
        in_channels = [64, 128, 256]
        num_classes = 10
        head_kwargs = {}
        reg_max = 16

        multihead_detection = MultiheadDetection(in_channels, num_classes, **head_kwargs)

        x_list = [
            torch.randn(2, in_channels[0], 10, 10),
            torch.randn(2, in_channels[1], 5, 5),
            torch.randn(2, in_channels[2], 3, 3),
        ]
        output = multihead_detection(x_list)

        assert len(output) == len(x_list)
        for o, x in zip(output, x_list):
            assert o[0].shape == (2, num_classes, x.shape[2], x.shape[3])
            assert o[1].shape == (2, reg_max, 4, x.shape[2], x.shape[3])
            assert o[2].shape == (2, 4, x.shape[2], x.shape[3])


class TestYOLOHeadModule:
    @pytest.fixture()
    def yolo_head(self) -> YOLOHeadModule:
        num_classes = 10
        cfg = {
            "csp_channels": [[320, 128, 128], [288, 192, 192], [384, 256, 256]],
            "aconv_channels": [[128, 96], [192, 128]],
            "concat_sources": [[-1, "N4"], [-1, "N3"]],
            "pre_upsample_concat_cfg": {"source": [-1, "B3"]},
            "csp_args": {"repeat_num": 3},
            "aux_cfg": {
                "sppelan_channels": [256, 256],
                "csp_channels": [[448, 192, 192], [320, 128, 128]],
            },
        }
        return YOLOHeadModule(num_classes=num_classes, **cfg)

    @pytest.fixture()
    def head_inputs(self) -> dict[int | str, Any]:
        return {
            0: torch.randn(1, 3, 640, 640),
            -1: torch.randn(1, 192, 40, 40),
            "B3": torch.randn(1, 128, 80, 80),
            "B4": torch.randn(1, 192, 40, 40),
            "B5": torch.randn(1, 256, 20, 20),
            "N3": torch.randn(1, 256, 20, 20),
            "N4": torch.randn(1, 192, 40, 40),
        }

    def test_forward(self, yolo_head: YOLOHeadModule, head_inputs: dict[int | str, Any]) -> None:
        main_preds, aux_preds = yolo_head(head_inputs)

        for main_pred, shape in zip(main_preds, [(80, 80), (40, 40), (20, 20)]):
            assert main_pred[0].shape == torch.Size([1, 10, *shape])
            assert main_pred[1].shape == torch.Size([1, 16, 4, *shape])
            assert main_pred[2].shape == torch.Size([1, 4, *shape])

        for aux_pred, shape in zip(aux_preds, [(80, 80), (40, 40), (20, 20)]):
            assert aux_pred[0].shape == torch.Size([1, 10, *shape])
            assert aux_pred[1].shape == torch.Size([1, 16, 4, *shape])
            assert aux_pred[2].shape == torch.Size([1, 4, *shape])

    def test_prepare_loss_inputs(
        self,
        yolo_head: YOLOHeadModule,
        head_inputs: dict[int | str, Any],
        fxt_det_data_entity,
    ) -> None:
        entity = deepcopy(fxt_det_data_entity[1])

        loss = yolo_head.prepare_loss_inputs(head_inputs, entity)

        assert "main_preds" in loss
        assert "aux_preds" in loss
        assert "targets" in loss

    def test_predict(self, yolo_head: YOLOHeadModule, head_inputs: dict[int | str, Any], fxt_det_data_entity) -> None:
        yolo_head.vec2box = Vec2Box(None, (640, 640), [8, 16, 32])
        entity = deepcopy(fxt_det_data_entity)[1]

        predictions = yolo_head.predict(head_inputs, entity, rescale=True)

        assert len(predictions) == 1
        assert isinstance(predictions[0], InstanceData)
        assert hasattr(predictions[0], "scores")
        assert hasattr(predictions[0], "bboxes")
        assert hasattr(predictions[0], "labels")

    def test_export(self, yolo_head: YOLOHeadModule, head_inputs: dict[int | str, Any]) -> None:
        yolo_head.vec2box = Vec2Box(None, (640, 640), [8, 16, 32])

        detection_results = yolo_head.export(head_inputs, None)

        assert len(detection_results) == 2
        assert isinstance(detection_results[0], torch.Tensor)
        assert isinstance(detection_results[1], torch.Tensor)

    def test_pad_bbox_labels(self, yolo_head: YOLOHeadModule) -> None:
        bboxes = [torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]), torch.tensor([[9, 10, 11, 12]])]
        labels = [torch.tensor([0, 1]), torch.tensor([2])]

        padded_bboxes, padded_labels = yolo_head.pad_bbox_labels(bboxes, labels)

        expected_padded_bboxes = torch.tensor(
            [
                [[1, 2, 3, 4], [5, 6, 7, 8]],
                [[9, 10, 11, 12], [0, 0, 0, 0]],
            ],
        )
        expected_padded_labels = torch.tensor(
            [
                [[0], [1]],
                [[2], [-1]],
            ],
        )

        assert torch.all(torch.eq(padded_bboxes, expected_padded_bboxes))
        assert torch.all(torch.eq(padded_labels, expected_padded_labels))
