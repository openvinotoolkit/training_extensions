# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest

from getitune.backend.lightning.exporter.base import ModelExporter
from getitune.backend.lightning.models.base import DataInputParams
from getitune.backend.lightning.models.segmentation.dino_v2_seg import DinoV2Seg


class TestDinoV2Seg:
    @pytest.fixture
    def fxt_dino_v2_seg(self) -> DinoV2Seg:
        return DinoV2Seg(
            label_info=10,
            model_name="dinov2-small-seg",
            data_input_params=DataInputParams((518, 518), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            pretrained=False,
        )

    def test_dino_v2_seg_init(self, fxt_dino_v2_seg):
        assert isinstance(fxt_dino_v2_seg, DinoV2Seg)
        assert fxt_dino_v2_seg.num_classes == 10

    def test_exporter(self, fxt_dino_v2_seg):
        exporter = fxt_dino_v2_seg._exporter
        assert isinstance(exporter, ModelExporter)
        assert exporter.data_input_params.input_size == (518, 518)

    def test_optimization_config(self, fxt_dino_v2_seg):
        config = fxt_dino_v2_seg._optimization_config
        assert isinstance(config, dict)
        assert "model_type" in config
        assert config["model_type"] == "transformer"
