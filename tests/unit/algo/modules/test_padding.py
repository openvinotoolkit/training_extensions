# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from otx.algo.modules.padding import build_padding_layer
from torch import nn


def test_build_padding_layer():
    cfg = {"type": "zero"}
    conv = build_padding_layer(cfg, padding=2)
    assert isinstance(conv, nn.ZeroPad2d)

    cfg = {"type": "reflect"}
    conv = build_padding_layer(cfg, padding=2)
    assert isinstance(conv, nn.ReflectionPad2d)

    cfg = {"type": "replicate"}
    conv = build_padding_layer(cfg, padding=2)
    assert isinstance(conv, nn.ReplicationPad2d)

    with pytest.raises(TypeError):
        build_padding_layer(None)

    with pytest.raises(KeyError, match='the cfg dict must contain the key "type"'):
        build_padding_layer({"cfg": 1})

    with pytest.raises(KeyError, match="Cannot find"):
        build_padding_layer({"type": "None"})
