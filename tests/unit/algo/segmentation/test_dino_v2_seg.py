# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import pytest
from otx.algo.segmentation.dino_v2_seg import DinoV2Seg


class TestDinoV2Seg:
    @pytest.fixture()
    def fxt_dino_v2_seg(self) -> DinoV2Seg:
        return DinoV2Seg(num_classes=10)

    def test_dino_v2_seg_init(self, fxt_dino_v2_seg):
        assert isinstance(fxt_dino_v2_seg, DinoV2Seg)
        assert fxt_dino_v2_seg.num_classes == 10

    def test_export_parameters(self, fxt_dino_v2_seg):
        parameters = fxt_dino_v2_seg._export_parameters
        assert isinstance(parameters, dict)
        assert "input_size" in parameters
        assert parameters["input_size"] == (1, 3, 560, 560)

    def test_optimization_config(self, fxt_dino_v2_seg):
        config = fxt_dino_v2_seg._optimization_config
        assert isinstance(config, dict)
        assert "model_type" in config
        assert config["model_type"] == "transformer"
