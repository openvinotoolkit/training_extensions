# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of getitune RTMDetInst architecture."""

import torch

from getitune.backend.lightning.models.base import DataInputParams
from getitune.backend.lightning.models.instance_segmentation.rtmdet_inst import RTMDetInst
from getitune.data.entity.sample import PredictionBatch


class TestRTMDetInst:
    def test_loss(self, fxt_instance_seg_batch):
        model = RTMDetInst(
            label_info=3,
            model_name="rtmdet_inst_tiny",
            data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            pretrained=False,
        )

        output = model(fxt_instance_seg_batch)
        assert "loss_cls" in output
        assert "loss_bbox" in output
        assert "loss_mask" in output

    def test_predict(self, fxt_instance_seg_batch):
        model = RTMDetInst(
            label_info=3,
            model_name="rtmdet_inst_tiny",
            data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            pretrained=False,
        )
        model.eval()
        output = model(fxt_instance_seg_batch)
        assert isinstance(output, PredictionBatch)

    def test_export(self):
        model = RTMDetInst(
            label_info=3,
            model_name="rtmdet_inst_tiny",
            data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            pretrained=False,
        )
        model.eval()
        output = model.forward_for_tracing(torch.randn(1, 3, 32, 32))
        assert len(output) == 3

    def test_export_parameters_disable_runtime_nms(self):
        model = RTMDetInst(
            label_info=3,
            model_name="rtmdet_inst_tiny",
            data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            pretrained=False,
        )

        assert model._export_parameters.nms_execute is False
