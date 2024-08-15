# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
# https://github.com/open-mmlab/mmcv/blob/main/tests/test_cnn/test_swish.py

import torch
from otx.algo.modules.activation import Swish
from torch.nn import functional


def test_swish():
    act = Swish()
    inputs = torch.randn(1, 3, 64, 64)
    expected_output = inputs * functional.sigmoid(inputs)
    output = act(inputs)
    # test output shape
    assert output.shape == expected_output.shape
    # test output value
    assert torch.equal(output, expected_output)
