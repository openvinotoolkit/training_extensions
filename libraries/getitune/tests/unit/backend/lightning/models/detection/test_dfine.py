# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Test of D-Fine."""

import pytest
import torch

from getitune.backend.lightning.models.base import DataInputParams
from getitune.backend.lightning.models.detection.d_fine import DFine
from getitune.data.entity.sample import PredictionBatch


class TestDFine:
    @pytest.fixture
    def fxt_model(self) -> DFine:
        return DFine(
            label_info=3,
            model_name="dfine_hgnetv2_x",
            data_input_params=DataInputParams((640, 640), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            pretrained=False,
        )

    def test_loss(self, fxt_model, fxt_detection_batch):
        fxt_model(fxt_detection_batch)

    def test_predict(self, fxt_model, fxt_detection_batch):
        fxt_model.eval()
        output = fxt_model(fxt_detection_batch)
        assert isinstance(output, PredictionBatch)

    def test_export(self, fxt_model):
        fxt_model.eval()
        output = fxt_model.forward_for_tracing(torch.randn(1, 3, 640, 640))
        assert len(output) == 3
