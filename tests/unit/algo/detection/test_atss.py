# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of OTX SSD architecture."""

from otx.algo.detection.atss import MobileNetV2ATSS
from otx.algo.utils.support_otx_v1 import OTXv1Helper
from otx.core.exporter.native import OTXModelExporter
from otx.core.types.export import TaskLevelExportParameters


class TestATSS:
    def test(self, mocker) -> None:
        model = MobileNetV2ATSS(2)
        mock_load_ckpt = mocker.patch.object(OTXv1Helper, "load_det_ckpt")
        model.load_from_otx_v1_ckpt({})
        mock_load_ckpt.assert_called_once_with({}, "model.model.")

        assert isinstance(model._export_parameters, TaskLevelExportParameters)
        assert isinstance(model._exporter, OTXModelExporter)
