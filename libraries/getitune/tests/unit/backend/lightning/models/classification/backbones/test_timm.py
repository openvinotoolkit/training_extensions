# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, cast

import pytest
import torch

from getitune.backend.lightning.models.classification.backbones.timm import TimmBackbone


class TestBackbone:
    @pytest.mark.parametrize(
        ("backbone_name", "expected_feature_dim"),
        [
            ("tf_efficientnetv2_s.in21k", 1280),
            ("vit_base_patch16_224", 768),
            ("swin_tiny_patch4_window7_224", 768),
        ],
    )
    def test_forward(self, backbone_name: str, expected_feature_dim: int) -> None:
        backbone = TimmBackbone(model_name=backbone_name)
        default_cfg = cast("dict[str, Any]", backbone.model.default_cfg)  # pyrefly: ignore[missing-attribute]
        _, h, w = default_cfg["input_size"]
        assert backbone(torch.randn(1, 3, h, w))[0].shape == torch.Size([expected_feature_dim])

    def test_num_features(self) -> None:
        backbone = TimmBackbone(model_name="samvit_base_patch16.sa1b")
        assert backbone.num_features == 256
