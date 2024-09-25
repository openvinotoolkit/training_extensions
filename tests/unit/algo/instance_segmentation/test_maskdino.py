# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of OTX MaskDINO architecture."""

import pytest
import torch
from otx.algo.instance_segmentation.maskdino import MaskDINOR50
from otx.algo.utils.support_otx_v1 import OTXv1Helper
from otx.core.data.entity.instance_segmentation import InstanceSegBatchPredEntity
from otx.core.types.export import TaskLevelExportParameters


class TestMaskDINO:
    def test_load_weights(self, mocker) -> None:
        model = MaskDINOR50(2, "maskdino_resnet_50")
        mock_load_ckpt = mocker.patch.object(OTXv1Helper, "load_iseg_ckpt")
        model.load_from_otx_v1_ckpt({})
        mock_load_ckpt.assert_called_once_with({}, "model.")

        assert isinstance(model._export_parameters, TaskLevelExportParameters)

    @pytest.mark.parametrize("model", [MaskDINOR50(3, "maskdino_resnet_50")])
    def test_loss(self, model, fxt_data_module):
        data = next(iter(fxt_data_module.train_dataloader()))
        data.images = torch.randn([2, 3, 32, 32])
        model(data)

    @pytest.mark.parametrize("model", [MaskDINOR50(3, "maskdino_resnet_50")])
    def test_predict(self, model, fxt_data_module):
        data = next(iter(fxt_data_module.train_dataloader()))
        data.images = [torch.randn(3, 32, 32), torch.randn(3, 48, 48)]
        model.eval()
        output = model(data)
        assert isinstance(output, InstanceSegBatchPredEntity)

    @pytest.mark.parametrize("model", [MaskDINOR50(3, "maskdino_resnet_50")])
    def test_export(self, model):
        model.eval()
        output = model.forward_for_tracing(torch.randn(1, 3, 32, 32))
        assert len(output) == 3
