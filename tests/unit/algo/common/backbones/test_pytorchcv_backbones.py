# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of pytorchcv backbones."""

from __future__ import annotations

import torch
from otx.algo.common.backbones.pytorchcv_backbones import (
    build_model_including_pytorchcv,
    multioutput_forward,
    replace_activation,
    replace_norm,
    train,
)
from torch import nn


class MockModule(nn.Module):
    def __init__(self):
        super().__init__()
        self._modules = {
            "linear1": nn.Linear(256, 256),
            "bn1": nn.BatchNorm2d(256),
            "activ1": nn.ReLU(),
            "linear2": nn.Linear(256, 32),
            "bn2": nn.BatchNorm2d(32),
            "activ2": nn.ReLU(),
            "linear3": nn.Linear(32, 10),
        }


def test_replace_activation() -> None:
    activation = nn.GELU
    model = MockModule()
    model = replace_activation(model, activation)
    assert isinstance(model._modules["activ1"], nn.GELU)
    assert isinstance(model._modules["activ2"], nn.GELU)

    activation = nn.SiLU
    model = replace_activation(model, activation)
    assert isinstance(model._modules["activ1"], nn.SiLU)
    assert isinstance(model._modules["activ2"], nn.SiLU)


def test_replace_norm(mocker) -> None:
    mocker.patch(
        "otx.algo.common.backbones.pytorchcv_backbones.build_norm_layer",
        return_value=[None, nn.BatchNorm1d(100)],
    )
    cfg = {"type": "BatchNorm1d"}
    model = MockModule()
    model = replace_norm(model, cfg)
    assert isinstance(model._modules["bn1"], nn.BatchNorm1d)
    assert isinstance(model._modules["bn2"], nn.BatchNorm1d)


def test_multioutput_forward() -> None:
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.out_indices = [1, 2]
            self.features = [
                nn.Linear(256, 256),
                nn.Linear(256, 128),
                nn.Linear(128, 64),
                nn.Linear(64, 32),
            ]
            self.verbose = True

    model = MockModel()
    out = multioutput_forward(model, torch.randn(3, 256))
    assert len(out) == 2
    assert out[0].shape == torch.Size([3, 128])
    assert out[1].shape == torch.Size([3, 64])


def test_train() -> None:
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.frozen_stages = 2
            self.features = [
                nn.Linear(256, 256),
                nn.Linear(256, 128),
                nn.Linear(128, 64),
                nn.Linear(64, 32),
            ]
            self.norm_eval = True

    model = MockModel()
    train(model)
    assert model.features[0].training is False
    assert model.features[1].training is False
    assert model.features[2].training is False
    assert model.features[3].training is True


def test_generate_backbones() -> None:
    cfg = {"type": "alexnet", "out_indices": [-1]}
    model = build_model_including_pytorchcv(cfg)

    assert "alexnet" in model.__class__.__name__.lower()
    assert model.out_indices == [-1]
