# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
# https://github.com/open-mmlab/mmcv/blob/main/tests/test_cnn/test_swish.py

from functools import partial

import pytest
import torch
from otx.algo.modules.activation import Swish, build_activation_layer
from torch import nn
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


def test_build_activation_layer():
    activation_layer = build_activation_layer(nn.PReLU)
    assert isinstance(activation_layer, nn.PReLU)

    activation_layer = build_activation_layer(nn.ReLU)
    assert isinstance(activation_layer, nn.ReLU)

    activation_layer = build_activation_layer(nn.LeakyReLU)
    assert isinstance(activation_layer, nn.LeakyReLU)

    activation_layer = build_activation_layer(Swish)
    assert isinstance(activation_layer, Swish)


def test_build_activation_layer_with_unsupported_activation():
    activation = nn.Softmax
    with pytest.raises(ValueError, match="Unsupported activation"):
        # softmax is not supported
        build_activation_layer(activation=activation)

    activation = partial(nn.Softmax)
    with pytest.raises(ValueError, match="Unsupported activation"):
        # softmax is not supported
        build_activation_layer(activation=activation)
