# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from functools import partial

import pytest
from otx.algo.modules.norm import build_norm_layer
from torch import nn


@pytest.mark.parametrize(
    ("normalization_callable", "expected_name"),
    [
        (nn.BatchNorm2d, "bn"),
        (nn.InstanceNorm2d, "in"),
        (nn.SyncBatchNorm, "bn"),
        (nn.LayerNorm, "ln"),
        (nn.GroupNorm, "gn"),
    ],
)
def test_build_norm_layer_with_nn_module(normalization_callable: type, expected_name: str) -> None:
    kwargs = {"num_groups": 1} if expected_name == "gn" else {}
    name, norm = build_norm_layer(normalization_callable, num_features=1, **kwargs)
    assert isinstance(norm, normalization_callable)
    assert name == expected_name


@pytest.mark.parametrize(
    ("normalization_callable", "expected_name"),
    [
        (nn.BatchNorm2d, "bn"),
        (nn.InstanceNorm2d, "in"),
        (nn.SyncBatchNorm, "bn"),
        (nn.LayerNorm, "ln"),
        (nn.GroupNorm, "gn"),
    ],
)
def test_build_norm_layer_with_partial(normalization_callable: type, expected_name: str) -> None:
    kwargs = {"num_groups": 1} if expected_name == "gn" else {}
    name, norm = build_norm_layer(partial(normalization_callable), num_features=1, **kwargs)
    assert isinstance(norm, normalization_callable)
    assert name == expected_name


@pytest.mark.parametrize(
    ("normalization_callable", "expected_name"),
    [
        (nn.BatchNorm2d, "bn"),
        (nn.InstanceNorm2d, "in"),
        (nn.SyncBatchNorm, "bn"),
        (nn.LayerNorm, "ln"),
        (nn.GroupNorm, "gn"),
    ],
)
def test_build_norm_layer_with_partial_build_norm_layer(normalization_callable: type, expected_name: str) -> None:
    kwargs = {"num_groups": 1} if expected_name == "gn" else {}
    name, norm = build_norm_layer(
        partial(build_norm_layer, normalization_callable, **kwargs),
        num_features=1,
        **kwargs,
    )
    assert isinstance(norm, normalization_callable)
    assert name == expected_name


@pytest.mark.parametrize(
    ("normalization_callable", "expected_name"),
    [
        (nn.BatchNorm2d, "bn"),
        (nn.InstanceNorm2d, "in"),
        (nn.SyncBatchNorm, "bn"),
        (nn.LayerNorm, "ln"),
        (nn.GroupNorm, "gn"),
    ],
)
def test_build_norm_layer_with_pre_assigned_module(normalization_callable: type, expected_name: str) -> None:
    kwargs = {"num_groups": 1} if expected_name == "gn" else {}
    name, norm = build_norm_layer(
        build_norm_layer(normalization_callable, num_features=1, **kwargs),
        num_features=1,
        **kwargs,
    )
    assert isinstance(norm, normalization_callable)
    assert name == expected_name


def test_build_norm_layer_with_invalid_module():
    with pytest.raises(TypeError):
        build_norm_layer(None, num_features=1)


def test_build_norm_layer_with_unsupported_module():
    with pytest.raises(ValueError, match="Unsupported normalization"):
        build_norm_layer(nn.LazyBatchNorm2d, num_features=1)
