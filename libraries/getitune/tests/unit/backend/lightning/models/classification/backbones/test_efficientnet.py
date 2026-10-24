# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch

from getitune.backend.lightning.models.classification.backbones.efficientnet import EfficientNetBackbone


class TestEfficientNet:
    @pytest.mark.parametrize(
        "model_name",
        [
            "efficientnet_b0",
            "efficientnet_b1",
            "efficientnet_b2",
            "efficientnet_b3",
            "efficientnet_b4",
            "efficientnet_b5",
            "efficientnet_b6",
            "efficientnet_b7",
            "efficientnet_b8",
        ],
    )
    def test_forward(self, model_name):
        model = EfficientNetBackbone(model_name)
        assert model(torch.randn(1, 3, 244, 244))[0].shape[-1] == 8
        assert model(torch.randn(1, 3, 244, 244))[0].shape[-2] == 8

    def test_set_input_size(self):
        input_size = (300, 300)
        model = EfficientNetBackbone("efficientnet_b0", input_size=input_size)
        assert model.in_size == input_size
