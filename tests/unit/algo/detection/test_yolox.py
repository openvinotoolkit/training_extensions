# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of OTX SSD architecture."""

import pytest
from otx.algo.detection.yolox import YoloX, YoloXTiny
from otx.algo.utils.support_otx_v1 import OTXv1Helper


class TestYOLOX:
    @pytest.mark.parametrize(
        "model",
        [
            YoloX(num_classes=2, variant="l"),
            YoloX(num_classes=2, variant="s"),
            YoloX(num_classes=2, variant="tiny"),
            YoloX(num_classes=2, variant="x"),
            YoloXTiny(num_classes=2),
        ],
    )
    def test(self, model, mocker) -> None:
        mock_load_ckpt = mocker.patch.object(OTXv1Helper, "load_det_ckpt")
        model.load_from_otx_v1_ckpt({})
        mock_load_ckpt.assert_called_once_with({}, "model.model.")

        assert isinstance(model._export_parameters, dict)
