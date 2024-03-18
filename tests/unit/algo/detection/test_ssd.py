# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of OTX SSD architecture."""

import pytest
from otx.algo.detection.ssd import SSD


class TestSSD:
    @pytest.fixture()
    def fxt_model(self) -> SSD:
        return SSD(num_classes=3, variant="mobilenetv2")

    def test_save_and_load_anchors(self, fxt_model) -> None:
        anchor_widths = fxt_model.model.bbox_head.anchor_generator.widths
        anchor_heights = fxt_model.model.bbox_head.anchor_generator.heights
        state_dict = fxt_model.state_dict()
        assert anchor_widths == state_dict["model.model.anchors"]["widths"]
        assert anchor_heights == state_dict["model.model.anchors"]["heights"]

        state_dict["model.model.anchors"]["widths"][0][0] = 40
        state_dict["model.model.anchors"]["heights"][0][0] = 50

        fxt_model.load_state_dict(state_dict)
        assert fxt_model.model.bbox_head.anchor_generator.widths[0][0] == 40
        assert fxt_model.model.bbox_head.anchor_generator.heights[0][0] == 50
