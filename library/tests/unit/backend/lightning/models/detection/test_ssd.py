# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of getitune SSD architecture."""

from pathlib import Path

import pytest
import torch
from lightning import Trainer

from getitune.backend.lightning.exporter.native import ModelExporter
from getitune.backend.lightning.models.base import DataInputParams
from getitune.backend.lightning.models.detection import SSD
from getitune.data.entity.sample import PredictionBatch
from getitune.types.export import TaskLevelExportParameters


class TestSSD:
    @pytest.fixture
    def fxt_model(self) -> SSD:
        return SSD(
            model_name="ssd_mobilenetv2",
            label_info=3,
            data_input_params=DataInputParams((320, 320), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            pretrained=False,
        )

    @pytest.fixture
    def fxt_checkpoint(self, fxt_model, tmpdir):
        trainer = Trainer(max_steps=0)

        # Directly set the anchor values in the model and hparams
        fxt_model.model.bbox_head.anchor_generator.widths[0][0] = 40
        fxt_model.model.bbox_head.anchor_generator.heights[0][0] = 50
        fxt_model.hparams["ssd_anchors"] = {
            "widths": fxt_model.model.bbox_head.anchor_generator.widths,
            "heights": fxt_model.model.bbox_head.anchor_generator.heights,
        }

        checkpoint_path = Path(tmpdir) / "checkpoint.ckpt"
        trainer.strategy._lightning_module = fxt_model
        trainer.save_checkpoint(checkpoint_path)

        return checkpoint_path

    def test_init(self, fxt_model):
        assert isinstance(fxt_model._export_parameters, TaskLevelExportParameters)
        assert isinstance(fxt_model._exporter, ModelExporter)

    def test_save_and_load_anchors(self, fxt_checkpoint) -> None:
        loaded_model = SSD.load_from_checkpoint(
            checkpoint_path=fxt_checkpoint, model_name="ssd_mobilenetv2", label_info=3, weights_only=True
        )

        assert loaded_model.model.bbox_head.anchor_generator.widths[0][0] == 40
        assert loaded_model.model.bbox_head.anchor_generator.heights[0][0] == 50

    def test_load_state_dict_pre_hook(self, fxt_model) -> None:
        prev_model = SSD(
            model_name="ssd_mobilenetv2",
            label_info=2,
            data_input_params=DataInputParams((320, 320), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            pretrained=False,
        )
        state_dict = prev_model.state_dict()
        fxt_model.model_classes = [1, 2, 3]
        fxt_model.ckpt_classes = [1, 2]
        fxt_model.load_state_dict_pre_hook(state_dict, "")
        keys = [
            key
            for key in prev_model.state_dict()
            if prev_model.state_dict()[key].shape != state_dict[key].shape
            or torch.all(prev_model.state_dict()[key] != state_dict[key])
        ]

        classification_layers = fxt_model._identify_classification_layers()

        for key in keys:
            assert key in classification_layers

    def test_loss(self, fxt_model, fxt_detection_batch):
        fxt_model.train()
        output = fxt_model(fxt_detection_batch)
        assert "loss_cls" in output
        assert "loss_bbox" in output

    def test_predict(self, fxt_model, fxt_detection_batch):
        fxt_model.eval()
        output = fxt_model(fxt_detection_batch)
        assert isinstance(output, PredictionBatch)

    def test_export(self, fxt_model):
        fxt_model.eval()
        output = fxt_model.forward_for_tracing(torch.randn(1, 3, 32, 32))
        assert len(output) == 2

        fxt_model.explain_mode = True
        output = fxt_model.forward_for_tracing(torch.randn(1, 3, 32, 32))
        assert len(output) == 4
