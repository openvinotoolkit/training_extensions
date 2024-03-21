# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of OTX SSD architecture."""

import pytest
from otx.algo.detection.atss import ATSS, ATSSR50FPN
from otx.algo.utils.support_otx_v1 import OTXv1Helper


class TestATSS:
    @pytest.mark.parametrize(
        "model",
        [
            ATSS(num_classes=2, variant="mobilenetv2"),
            ATSS(num_classes=2, variant="r50_fpn"),
            ATSS(num_classes=2, variant="resnext101"),
            ATSSR50FPN(num_classes=2),
        ],
    )
    def test(self, model, mocker) -> None:
        mock_load_ckpt = mocker.patch.object(OTXv1Helper, "load_det_ckpt")
        model.load_from_otx_v1_ckpt({})
        mock_load_ckpt.assert_called_once_with({}, "model.model.")

        assert isinstance(model._export_parameters, dict)
