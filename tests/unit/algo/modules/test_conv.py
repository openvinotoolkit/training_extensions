# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from otx.algo.modules.conv import build_conv_layer
from torch import nn


def test_build_conv_layer():
    cfg = {"type": "Conv1d"}
    conv = build_conv_layer(cfg, in_channels=1, out_channels=1, kernel_size=1)
    assert isinstance(conv, nn.Conv1d)

    cfg = {"type": "Conv2d"}
    conv = build_conv_layer(cfg, in_channels=1, out_channels=1, kernel_size=1)
    assert isinstance(conv, nn.Conv2d)

    cfg = {"type": "Conv3d"}
    conv = build_conv_layer(cfg, in_channels=1, out_channels=1, kernel_size=1)
    assert isinstance(conv, nn.Conv3d)

    cfg = {"type": "Conv"}
    conv = build_conv_layer(cfg, in_channels=1, out_channels=1, kernel_size=1)
    assert isinstance(conv, nn.Conv2d)

    with pytest.raises(TypeError):
        build_conv_layer(None)

    with pytest.raises(KeyError, match='the cfg dict must contain the key "type"'):
        build_conv_layer({"cfg": 1})

    with pytest.raises(KeyError, match="Cannot find"):
        build_conv_layer({"type": "None"})
