# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of OTX MonoDETR3D architecture."""

import pytest
import torch
from otx.algo.object_detection_3d.monodetr3d import MonoDETR3D
from otx.core.data.entity.object_detection_3d import Det3DBatchDataEntity
from otx.core.exporter.detection_3d import OTXObjectDetection3DExporter
from otx.core.types.export import TaskLevelExportParameters


class TestMonoDETR3D:
    @pytest.fixture()
    def model(self):
        return MonoDETR3D(model_name="monodetr_50", label_info=2, input_size=(1280, 384))

    def test_init(self, model) -> None:
        assert isinstance(model._export_parameters, TaskLevelExportParameters)
        assert isinstance(model._exporter, OTXObjectDetection3DExporter)

    def test_loss(self, model, fxt_data_module_3d):
        data = next(iter(fxt_data_module_3d.train_dataloader()))
        output = model(data)
        assert "loss_ce" in output
        assert "loss_bbox" in output
        assert "loss_center" in output
        assert "loss_center_aux_1" in output
        for loss in output.values():
            assert loss is not None
            assert isinstance(loss, torch.Tensor)

    def test_predict(self, model, fxt_data_module_3d):
        data = next(iter(fxt_data_module_3d.train_dataloader()))
        model.eval()
        output = model(data)
        assert isinstance(output, Det3DBatchDataEntity)

    def test_export(self, model):
        model.eval()
        output = model.forward_for_tracing(
            torch.randn(1, 3, 384, 1280),
            torch.randn(1, 3, 4),
            torch.tensor([[1280, 384]]),
        )
        assert isinstance(output, dict)
        assert len(output) == 5
        assert list(output.keys()) == ["scores", "boxes_3d", "size_3d", "depth", "heading_angle"]
