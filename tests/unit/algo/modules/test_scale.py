# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
# https://github.com/open-mmlab/mmcv/blob/main/tests/test_cnn/test_scale.py
import torch
from otx.algo.modules.scale import Scale


def test_scale():
    # test default scale
    scale = Scale()
    assert scale.scale.data == 1.0
    assert scale.scale.dtype == torch.float
    x = torch.rand(1, 3, 64, 64)
    output = scale(x)
    assert output.shape == (1, 3, 64, 64)

    # test given scale
    scale = Scale(10.0)
    assert scale.scale.data == 10.0
    assert scale.scale.dtype == torch.float
    x = torch.rand(1, 3, 64, 64)
    output = scale(x)
    assert output.shape == (1, 3, 64, 64)
