# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of OTX SSD architecture."""

import pytest
from otx.algo.detection.atss import ATSS
from otx.algo.utils.support_otx_v1 import OTXv1Helper
from otx.core.exporter.native import OTXModelExporter
from otx.core.types.export import TaskLevelExportParameters


class TestATSS:
    @pytest.mark.parametrize(("label_info", "variant"), [(2, "mobilenetv2"), (2, "resnext101")])
    def test(self, label_info, variant, mocker) -> None:
        model = ATSS(label_info, variant)
        mock_load_ckpt = mocker.patch.object(OTXv1Helper, "load_det_ckpt")
        model.load_from_otx_v1_ckpt({})
        mock_load_ckpt.assert_called_once_with({}, "model.model.")

        assert isinstance(model._export_parameters, TaskLevelExportParameters)
        assert isinstance(model._exporter, OTXModelExporter)
