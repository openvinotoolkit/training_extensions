# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of OTX SSD architecture."""

import pytest
from otx.algo.detection.rtmdet import RTMDet
from otx.algo.utils.support_otx_v1 import OTXv1Helper


class TestRTMDet:
    @pytest.fixture()
    def fxt_model(self) -> RTMDet:
        return RTMDet(num_classes=3, variant="tiny")

    def test(self, fxt_model, mocker) -> None:
        mock_load_ckpt = mocker.patch.object(OTXv1Helper, "load_det_ckpt")
        fxt_model.load_from_otx_v1_ckpt({})
        mock_load_ckpt.assert_called_once_with({}, "model.model.")

        assert isinstance(fxt_model._export_parameters, dict)
