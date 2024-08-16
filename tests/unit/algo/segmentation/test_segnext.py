# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import pytest
from otx.algo.segmentation.segnext import SegNext
from otx.algo.utils.support_otx_v1 import OTXv1Helper


class TestSegNext:
    @pytest.fixture()
    def fxt_segnext(self) -> SegNext:
        return SegNext(10, model_version="segnext_base", input_size=(512, 512))

    def test_segnext_init(self, fxt_segnext):
        assert isinstance(fxt_segnext, SegNext)
        assert fxt_segnext.num_classes == 10

    def test_load_from_otx_v1_ckpt(self, fxt_segnext, mocker):
        mock_load_ckpt = mocker.patch.object(OTXv1Helper, "load_seg_segnext_ckpt")
        fxt_segnext.load_from_otx_v1_ckpt({})
        mock_load_ckpt.assert_called_once_with({}, "model.model.")

    def test_optimization_config(self, fxt_segnext):
        config = fxt_segnext._optimization_config
        assert isinstance(config, dict)
        assert "ignored_scope" in config
        assert isinstance(config["ignored_scope"], dict)
        assert "patterns" in config["ignored_scope"]
        assert isinstance(config["ignored_scope"]["patterns"], list)
        assert "types" in config["ignored_scope"]
        assert isinstance(config["ignored_scope"]["types"], list)
