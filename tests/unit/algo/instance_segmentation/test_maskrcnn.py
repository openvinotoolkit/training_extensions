# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of OTX MaskRCNN architecture."""

import pytest
import torch
from otx.algo.instance_segmentation.maskrcnn import MaskRCNNEfficientNet, MaskRCNNResNet50, MaskRCNNSwinT
from otx.algo.instance_segmentation.maskrcnn_tv import TVMaskRCNNR50
from otx.algo.utils.support_otx_v1 import OTXv1Helper
from otx.core.data.entity.instance_segmentation import InstanceSegBatchPredEntity
from otx.core.types.export import TaskLevelExportParameters


class TestMaskRCNN:
    def test_load_weights(self, mocker) -> None:
        model = MaskRCNNResNet50(2)
        mock_load_ckpt = mocker.patch.object(OTXv1Helper, "load_iseg_ckpt")
        model.load_from_otx_v1_ckpt({})
        mock_load_ckpt.assert_called_once_with({}, "model.")

        assert isinstance(model._export_parameters, TaskLevelExportParameters)

    @pytest.mark.parametrize(
        "model",
        [MaskRCNNResNet50(3), MaskRCNNEfficientNet(3), MaskRCNNSwinT(3), TVMaskRCNNR50(3)],
    )
    def test_loss(self, model, fxt_data_module):
        data = next(iter(fxt_data_module.train_dataloader()))
        data.images = torch.randn([2, 3, 32, 32])

        output = model(data)
        if isinstance(model, TVMaskRCNNR50):
            assert "loss_classifier" in output
            assert "loss_box_reg" in output
            assert "loss_mask" in output
            assert "loss_objectness" in output
            assert "loss_rpn_box_reg" in output
        else:
            assert "loss_cls" in output
            assert "loss_bbox" in output
            assert "loss_mask" in output
            assert "loss_rpn_cls" in output
            assert "loss_rpn_bbox" in output

    @pytest.mark.parametrize(
        "model",
        [MaskRCNNResNet50(3), MaskRCNNEfficientNet(3), MaskRCNNSwinT(3), TVMaskRCNNR50(3)],
    )
    def test_predict(self, model, fxt_data_module):
        data = next(iter(fxt_data_module.train_dataloader()))
        data.images = [torch.randn(3, 32, 32), torch.randn(3, 48, 48)]
        model.eval()
        output = model(data)
        assert isinstance(output, InstanceSegBatchPredEntity)

    @pytest.mark.parametrize(
        "model",
        [MaskRCNNResNet50(3), MaskRCNNEfficientNet(3), MaskRCNNSwinT(3), TVMaskRCNNR50(3)],
    )
    def test_export(self, model):
        model.eval()
        output = model.forward_for_tracing(torch.randn(1, 3, 32, 32))
        assert len(output) == 3

        # TODO(Eugene): Explain should return proper output.
        # After enabling explain for maskrcnn, below codes shuold be passed
        # model.explain_mode = True  # noqa: ERA001
        # output = model.forward_for_tracing(torch.randn(1, 3, 32, 32))  # noqa: ERA001
        # assert len(output) == 5  # noqa: ERA001
