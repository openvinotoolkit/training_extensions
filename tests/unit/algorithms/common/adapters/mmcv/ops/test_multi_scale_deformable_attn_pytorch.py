"""Test for otx.algorithms.common.adapters.mmcv.ops.multi_scale_deformable_attn_pytorch."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch

from otx.algorithms.common.adapters.mmcv.ops import multi_scale_deformable_attn_pytorch
from tests.test_suite.e2e_test_system import e2e_pytest_unit


@e2e_pytest_unit
def test_multi_scale_deformable_attn_pytorch():
    value = torch.randn([1, 22223, 8, 32])
    value_spatial_shapes = torch.tensor([[100, 167], [50, 84], [25, 42], [13, 21]])
    sampling_locations = torch.randn([1, 2223, 8, 4, 4, 2])
    attention_weights = torch.randn([1, 2223, 8, 4, 4])

    out = multi_scale_deformable_attn_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights)
    assert out.shape == torch.Size([1, 2223, 256])
