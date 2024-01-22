# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import torch
from otx.algo.visual_prompting.utils.layer_norm_2d import LayerNorm2d


class TestLayerNorm2d:
    def test_forward(self) -> None:
        """Test forward."""
        layer_norm_2d = LayerNorm2d(num_channels=2)

        assert torch.all(layer_norm_2d.weight == torch.ones(2))
        assert torch.all(layer_norm_2d.bias == torch.zeros(2))

        inputs = torch.arange(32, dtype=torch.float32).view(1, 2, 4, 4)

        result = layer_norm_2d(inputs)

        assert result.shape == inputs.shape
        assert torch.all(result == torch.cat((torch.ones(1, 1, 4, 4) * (-1), torch.ones(1, 1, 4, 4)), dim=1))
