# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of OTX SSD architecture."""

from pathlib import Path

import pytest
from lightning import Trainer
from otx.algo.detection.ssd import SSD


class TestSSD:
    @pytest.fixture()
    def fxt_model(self) -> SSD:
        return SSD(num_classes=3, variant="mobilenetv2")

    @pytest.fixture()
    def fxt_checkpoint(self, fxt_model, fxt_data_module, tmpdir, monkeypatch: pytest.MonkeyPatch):
        trainer = Trainer(max_steps=0)

        monkeypatch.setattr(trainer.strategy, "_lightning_module", fxt_model)
        monkeypatch.setattr(trainer, "datamodule", fxt_data_module)
        monkeypatch.setattr(fxt_model, "_trainer", trainer)
        fxt_model.setup("fit")

        fxt_model.hparams["ssd_anchors"]["widths"][0][0] = 40
        fxt_model.hparams["ssd_anchors"]["heights"][0][0] = 50

        checkpoint_path = Path(tmpdir) / "checkpoint.ckpt"
        trainer.save_checkpoint(checkpoint_path)

        return checkpoint_path

    def test_save_and_load_anchors(self, fxt_checkpoint) -> None:
        loaded_model = SSD.load_from_checkpoint(checkpoint_path=fxt_checkpoint)

        assert loaded_model.model.bbox_head.anchor_generator.widths[0][0] == 40
        assert loaded_model.model.bbox_head.anchor_generator.heights[0][0] == 50
