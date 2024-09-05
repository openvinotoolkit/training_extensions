# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of YOLONeck architecture."""

import pytest
import torch
from otx.algo.detection.necks.yolo_neck import YOLONeckModule


class TestYOLONeckModule:
    @pytest.fixture()
    def yolo_neck(self) -> YOLONeckModule:
        cfg = {
            "elan_channels": [
                {"type": "SPPELAN", "args": {"in_channels": 256, "out_channels": 256}, "tags": "N3"},
                {
                    "type": "RepNCSPELAN",
                    "args": {"in_channels": 448, "out_channels": 192, "part_channels": 192},
                    "tags": "N4",
                },
            ],
            "concat_sources": [[-1, "B4"]],
            "csp_args": {"repeat_num": 3},
        }
        return YOLONeckModule(**cfg)

    def test_forward(self, yolo_neck) -> None:
        inputs = {
            0: torch.randn([1, 3, 640, 640]),
            -1: torch.randn([1, 256, 20, 20]),
            "B3": torch.randn([1, 128, 80, 80]),
            "B4": torch.randn([1, 192, 40, 40]),
            "B5": torch.randn([1, 256, 20, 20]),
        }

        results = yolo_neck(inputs)

        assert isinstance(results, dict)
        assert results[0].shape == torch.Size([1, 3, 640, 640])
        assert results[-1].shape == torch.Size([1, 192, 40, 40])
        assert results["B3"].shape == torch.Size([1, 128, 80, 80])
        assert results["B4"].shape == torch.Size([1, 192, 40, 40])
        assert results["B5"].shape == torch.Size([1, 256, 20, 20])
        assert results["N3"].shape == torch.Size([1, 256, 20, 20])
        assert results["N4"].shape == torch.Size([1, 192, 40, 40])
