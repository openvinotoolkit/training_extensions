# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from otx.algo.classification.backbones.mobilenet_v3 import OTXMobileNetV3


class TestOTXMobileNetV3:
    def test_forward(self):
        model = OTXMobileNetV3()
        assert model(torch.randn(1, 3, 244, 244))[0].shape == torch.Size([1, 960, 8, 8])

    def test_glob_feature_vector(self):
        model = OTXMobileNetV3()
        assert model._glob_feature_vector(torch.randn([1, 960, 8, 8]), "avg").shape == torch.Size([1, 960])
        assert model._glob_feature_vector(torch.randn([1, 960, 8, 8]), "max").shape == torch.Size([1, 960])
        assert model._glob_feature_vector(torch.randn([1, 960, 8, 8]), "avg+max").shape == torch.Size([1, 960])
        assert model._glob_feature_vector(torch.randn([1, 960, 8, 8]), "avg", reduce_dims=False).shape == torch.Size(
            [1, 960, 1, 1],
        )
