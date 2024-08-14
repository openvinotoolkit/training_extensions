# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest
from otx.algo.segmentation.dino_v2_seg import DinoV2Seg
from otx.core.exporter.base import OTXModelExporter


class TestDinoV2Seg:
    @pytest.fixture(scope="class")
    def fxt_dino_v2_seg(self) -> DinoV2Seg:
        return DinoV2Seg(label_info=10, model_version="dinov2_vits14", input_size=(560, 560))

    def test_dino_v2_seg_init(self, fxt_dino_v2_seg):
        assert isinstance(fxt_dino_v2_seg, DinoV2Seg)
        assert fxt_dino_v2_seg.num_classes == 10

    def test_exporter(self, fxt_dino_v2_seg):
        exporter = fxt_dino_v2_seg._exporter
        assert isinstance(exporter, OTXModelExporter)
        assert exporter.input_size == (1, 3, 560, 560)

    def test_optimization_config(self, fxt_dino_v2_seg):
        config = fxt_dino_v2_seg._optimization_config
        assert isinstance(config, dict)
        assert "model_type" in config
        assert config["model_type"] == "transformer"
