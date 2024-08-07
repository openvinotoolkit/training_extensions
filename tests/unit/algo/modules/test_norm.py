# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from otx.algo.modules.norm import build_norm_layer
from torch import nn


def test_build_norm_layer():
    cfg = {"type": "BN"}
    name, norm = build_norm_layer(cfg, num_features=1)
    assert isinstance(norm, nn.BatchNorm2d)
    assert name == "bn"

    cfg = {"type": "IN"}
    name, norm = build_norm_layer(cfg, num_features=1)
    assert isinstance(norm, nn.InstanceNorm2d)
    assert name == "in"

    cfg = {"type": "SyncBN"}
    name, norm = build_norm_layer(cfg, num_features=1)
    assert isinstance(norm, nn.SyncBatchNorm)
    assert name == "bn"

    cfg = {"type": "LN"}
    name, norm = build_norm_layer(cfg, num_features=1)
    assert isinstance(norm, nn.LayerNorm)
    assert name == "ln"

    with pytest.raises(TypeError):
        build_norm_layer(None, num_features=1)

    with pytest.raises(KeyError, match='the cfg dict must contain the key "type"'):
        build_norm_layer({"cfg": 1}, num_features=1)

    with pytest.raises(KeyError, match="Cannot find"):
        build_norm_layer({"type": "None"}, num_features=1)
