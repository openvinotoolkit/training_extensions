# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for instance segmentation model module."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
from otx.core.model.module.instance_segmentation import OTXInstanceSegLitModule


class TestOTXInstanceSegModule:
    @pytest.fixture()
    def fxt_model_ckpt(self) -> dict[str, torch.Tensor]:
        return {
            "model.model.backbone.1.weight": torch.randn(3, 10),
            "model.model.backbone.1.bias": torch.randn(3, 10),
            "model.model.head.1.weight": torch.randn(10, 2),
            "model.model.head.1.bias": torch.randn(10, 2),
        }

    @pytest.fixture()
    def fxt_model(self) -> OTXInstanceSegLitModule:
        return OTXInstanceSegLitModule(
            otx_model=MagicMock(spec=OTXInstanceSegLitModule),
            optimizer=MagicMock,
            scheduler=MagicMock,
            torch_compile=False,
        )

    def test_load_from_prev_otx_ckpt(self, fxt_model, fxt_model_ckpt) -> None:
        ckpt_otx_v1 = {
            "backbone.1.weight": torch.randn(3, 10),
            "backbone.1.bias": torch.randn(3, 10),
            "head.1.weight": torch.randn(10, 2),
            "head.1.bias": torch.randn(10, 2),
            "ema_model_t.1": torch.randn(3, 10),
        }
        converted_ckpt = fxt_model._load_from_prev_otx_ckpt(ckpt_otx_v1)

        assert fxt_model_ckpt.keys() == converted_ckpt.keys()
        for src_value, dst_value in zip(converted_ckpt.values(), fxt_model_ckpt.values()):
            assert src_value.shape == dst_value.shape
