# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of getitune MaskRCNN architecture."""

import pytest
import torch

from getitune.backend.lightning.models.base import DataInputParams
from getitune.backend.lightning.models.instance_segmentation.maskrcnn import MaskRCNN
from getitune.backend.lightning.models.instance_segmentation.maskrcnn_tv import MaskRCNNTV
from getitune.data.entity.sample import PredictionBatch
from getitune.types.export import TaskLevelExportParameters


class TestMaskRCNN:
    @pytest.fixture(
        params=[
            ("MaskRCNN", "maskrcnn_efficientnet_b2b"),
            ("MaskRCNN", "maskrcnn_swin_tiny"),
            ("MaskRCNNTV", "maskrcnn_resnet_50"),
        ],
    )
    def fxt_model(self, request):
        cls_name, model_name = request.param
        cls = MaskRCNN if cls_name == "MaskRCNN" else MaskRCNNTV
        return cls(
            label_info=3,
            model_name=model_name,
            data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            pretrained=False,
        )

    def test_load_weights(self, mocker) -> None:
        model = MaskRCNN(
            label_info=2,
            model_name="maskrcnn_efficientnet_b2b",
            data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            pretrained=False,
        )

        assert isinstance(model._export_parameters, TaskLevelExportParameters)

    def test_loss(self, fxt_model, fxt_instance_seg_batch):
        data = fxt_instance_seg_batch

        output = fxt_model(data)
        if fxt_model.model_name == "maskrcnn_resnet_50" and isinstance(fxt_model, MaskRCNNTV):
            assert "loss_classifier" in output
            assert "loss_box_reg" in output
            assert "loss_mask" in output
            assert "loss_objectness" in output
            assert "loss_rpn_box_reg" in output
        else:
            assert "loss_cls" in output
            assert "loss_bbox" in output
            assert "loss_mask" in output
            assert "loss_cls_rpn" in output
            assert "loss_bbox_rpn" in output

    def test_predict(self, fxt_model, fxt_instance_seg_batch):
        data = fxt_instance_seg_batch
        data.images = [torch.randn(3, 32, 32), torch.randn(3, 48, 48)]
        fxt_model.eval()
        output = fxt_model(data)
        assert isinstance(output, PredictionBatch)

    def test_export(self, fxt_model):
        fxt_model.eval()
        output = fxt_model.forward_for_tracing(torch.randn(1, 3, 32, 32))
        assert len(output) == 3

    def test_export_parameters_always_disable_runtime_nms(self, fxt_model):
        assert fxt_model._export_parameters.nms_execute is False

    def test_maskrcnn_tv_optimization_config_excludes_mask_roi_pool(self) -> None:
        """``MaskRCNNTV`` PTQ config must exclude the ``mask_roi_pool`` scatter subgraph.

        NNCF calibration on small datasets can leave scatter/slice nodes inside
        ``roi_heads.mask_roi_pool`` without statistics, raising
        ``InternalError: Statistics were not collected for the node
        __module.model.roi_heads.mask_roi_pool/aten::scatter/Slice_3``.
        We narrow the ignored scope to the failing scatter subgraph only, so
        the conv-dominated parts of the mask branch remain quantized.
        """
        model = MaskRCNNTV(
            label_info=3,
            model_name="maskrcnn_resnet_50",
            data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            pretrained=False,
        )
        cfg = model._optimization_config
        assert "ignored_scope" in cfg
        ignored = cfg["ignored_scope"]
        assert "patterns" in ignored
        patterns = ignored["patterns"]
        assert any("mask_roi_pool" in p and "scatter" in p for p in patterns), patterns
        assert ignored.get("validate") is False
