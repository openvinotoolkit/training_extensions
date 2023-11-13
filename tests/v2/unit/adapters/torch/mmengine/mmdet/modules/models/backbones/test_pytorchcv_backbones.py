"""Unit tests of otx/v2/adapters/torch/mmengine/mmdet/modules/backbones/test_pytorchcv_backbones.py."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn

from otx.v2.adapters.torch.mmengine.mmdet.modules.models.backbones.pytorchcv_backbones import (
    replace_activation,
    replace_norm,
    multioutput_forward,
    train,
    init_weights,
    generate_backbones,
)


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


def test_replace_activation():
    activation_cfg = {"type": "GELU"}
    model = MockModule()
    model = replace_activation(model, activation_cfg)
    assert isinstance(model._modules["activ1"], nn.GELU)
    assert isinstance(model._modules["activ2"], nn.GELU)

    activation_cfg = {"type": "torch_swish"}
    model = replace_activation(model, activation_cfg)
    assert isinstance(model._modules["activ1"], nn.SiLU)
    assert isinstance(model._modules["activ2"], nn.SiLU)


def test_replace_norm(mocker):
    mocker.patch(
        "otx.v2.adapters.torch.mmengine.mmdet.modules.models.backbones.pytorchcv_backbones.build_norm_layer",
        return_value=[None, nn.BatchNorm1d(100)],
    )
    cfg = {"type": "BatchNorm1d"}
    model = MockModule()
    model = replace_norm(model, cfg)
    assert isinstance(model._modules["bn1"], nn.BatchNorm1d)
    assert isinstance(model._modules["bn2"], nn.BatchNorm1d)


def test_multioutput_forward():
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


def test_train():
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


def test_generate_backbones(mocker):
    modules: list[nn.Module] = []

    def mock_register_module(name, module):
        modules.append([name, module])

    mocker.patch(
        "mmdet.registry.MODELS.register_module",
        side_effect=mock_register_module,
    )

    generate_backbones()
    alexnet = modules[0][1](out_indices=[-1])
    assert len(modules) == 857
    assert alexnet.out_indices == [-1]
