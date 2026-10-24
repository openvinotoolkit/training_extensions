# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import pytest

from getitune.backend.lightning.models.base import DataInputParams
from getitune.backend.lightning.models.segmentation.segnext import SegNext


class TestSegNext:
    @pytest.fixture
    def fxt_segnext(self) -> SegNext:
        return SegNext(
            10,
            model_name="segnext_base",
            data_input_params=DataInputParams((518, 518), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            pretrained=False,
        )

    def test_segnext_init(self, fxt_segnext):
        assert isinstance(fxt_segnext, SegNext)
        assert fxt_segnext.num_classes == 10

    def test_optimization_config(self, fxt_segnext):
        config = fxt_segnext._optimization_config
        assert isinstance(config, dict)
        assert "ignored_scope" in config
        assert isinstance(config["ignored_scope"], dict)
        assert "patterns" in config["ignored_scope"]
        assert isinstance(config["ignored_scope"]["patterns"], list)
        assert "types" in config["ignored_scope"]
        assert isinstance(config["ignored_scope"]["types"], list)
