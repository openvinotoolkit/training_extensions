# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of OTX MaskDINO architecture."""

import pytest
import torch
from otx.algo.instance_segmentation.maskdino import MaskDINOR50
from otx.core.data.entity.instance_segmentation import InstanceSegBatchPredEntity


class TestMaskDINO:
    @pytest.mark.parametrize("model", [MaskDINOR50(3, "maskdino_resnet_50")])
    def test_loss(self, model, fxt_data_module):
        data = next(iter(fxt_data_module.train_dataloader()))
        data.images = torch.randn([2, 3, 320, 320])
        data.masks = [torch.ones((len(masks), 320, 320)) for masks in data.masks]
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
