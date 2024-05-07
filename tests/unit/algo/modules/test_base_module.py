# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
# https://github.com/open-mmlab/mmengine/blob/main/tests/test_model/test_base_module.py
from unittest.mock import Mock

import pytest
import torch
from otx.algo.modules.base_module import BaseModule, ModuleDict, ModuleList, Sequential
from torch import nn
from torch.nn.init import constant_


class FooConv1d(BaseModule):
    def __init__(self, init_cfg=None):
        super().__init__(init_cfg)
        self.conv1d = nn.Conv1d(4, 1, 4)

    def forward(self, x):
        return self.conv1d(x)


class FooConv2d(BaseModule):
    def __init__(self, init_cfg=None):
        super().__init__(init_cfg)
        self.conv2d = nn.Conv2d(3, 1, 3)

    def forward(self, x):
        return self.conv2d(x)


class FooLinear(BaseModule):
    def __init__(self, init_cfg=None):
        super().__init__(init_cfg)
        self.linear = nn.Linear(3, 4)

    def forward(self, x):
        return self.linear(x)


class FooLinearConv1d(BaseModule):
    def __init__(self, linear=None, conv1d=None, init_cfg=None):
        super().__init__(init_cfg)
        self.linear = linear
        self.conv1d = conv1d

    def forward(self, x):
        x = self.linear(x)
        return self.conv1d(x)


class FooModel(BaseModule):
    def __init__(self, component1=None, component2=None, component3=None, component4=None, init_cfg=None) -> None:
        super().__init__(init_cfg)
        self.component1 = component1
        self.component2 = component2
        self.component3 = component3
        self.component4 = component4

        # its type is not BaseModule, it can be initialized
        # with "override" key.
        self.reg = nn.Linear(3, 4)


class TestBaseModule:
    @pytest.fixture()
    def fxt_model(self) -> FooModel:
        return FooModel(
            component1=FooConv1d(),
            component2=FooConv2d(),
            component3=FooLinear(),
            component4=FooLinearConv1d(linear=FooLinear(), conv1d=FooConv1d()),
            init_cfg=[
                {"type": "Constant", "val": 1, "bias": 2, "layer": "Linear"},
                {"type": "Constant", "val": 3, "bias": 4, "layer": "Conv1d"},
                {"type": "Constant", "val": 5, "bias": 6, "layer": "Conv2d"},
            ],
        )

    def test_init_weights(self, fxt_model, tmp_path):
        """
        Config
        model (FooModel, Linear: weight=1, bias=2, Conv1d: weight=3, bias=4,
                        Conv2d: weight=5, bias=6)
        ├──component1 (FooConv1d)
        ├──component2 (FooConv2d)
        ├──component3 (FooLinear)
        ├──component4 (FooLinearConv1d)
            ├──linear (FooLinear)
            ├──conv1d (FooConv1d)
        ├──reg (nn.Linear)
        Parameters after initialization
        model (FooModel)
        ├──component1 (FooConv1d, weight=3, bias=4)
        ├──component2 (FooConv2d, weight=5, bias=6)
        ├──component3 (FooLinear, weight=1, bias=2)
        ├──component4 (FooLinearConv1d)
            ├──linear (FooLinear, weight=1, bias=2)
            ├──conv1d (FooConv1d, weight=3, bias=4)
        ├──reg (nn.Linear, weight=1, bias=2)
        """
        fxt_model.init_weights()

        assert torch.equal(
            fxt_model.component1.conv1d.weight,
            torch.full(fxt_model.component1.conv1d.weight.shape, 3.0),
        )
        assert torch.equal(fxt_model.component1.conv1d.bias, torch.full(fxt_model.component1.conv1d.bias.shape, 4.0))
        assert torch.equal(
            fxt_model.component2.conv2d.weight,
            torch.full(fxt_model.component2.conv2d.weight.shape, 5.0),
        )
        assert torch.equal(fxt_model.component2.conv2d.bias, torch.full(fxt_model.component2.conv2d.bias.shape, 6.0))
        assert torch.equal(
            fxt_model.component3.linear.weight,
            torch.full(fxt_model.component3.linear.weight.shape, 1.0),
        )
        assert torch.equal(fxt_model.component3.linear.bias, torch.full(fxt_model.component3.linear.bias.shape, 2.0))
        assert torch.equal(
            fxt_model.component4.linear.linear.weight,
            torch.full(fxt_model.component4.linear.linear.weight.shape, 1.0),
        )
        assert torch.equal(
            fxt_model.component4.linear.linear.bias,
            torch.full(fxt_model.component4.linear.linear.bias.shape, 2.0),
        )
        assert torch.equal(
            fxt_model.component4.conv1d.conv1d.weight,
            torch.full(fxt_model.component4.conv1d.conv1d.weight.shape, 3.0),
        )
        assert torch.equal(
            fxt_model.component4.conv1d.conv1d.bias,
            torch.full(fxt_model.component4.conv1d.conv1d.bias.shape, 4.0),
        )
        assert torch.equal(fxt_model.reg.weight, torch.full(fxt_model.reg.weight.shape, 1.0))
        assert torch.equal(fxt_model.reg.bias, torch.full(fxt_model.reg.bias.shape, 2.0))

        # Test build model from Pretrained weights

        class CustomLinear(BaseModule):
            def __init__(self, init_cfg=None):
                super().__init__(init_cfg)
                self.linear = nn.Linear(1, 1)

            def init_weights(self) -> None:
                constant_(self.linear.weight, 1)
                constant_(self.linear.bias, 2)

        class PratrainedModel(FooModel):
            def __init__(
                self,
                component1=None,
                component2=None,
                component3=None,
                component4=None,
                init_cfg=None,
            ) -> None:
                super().__init__(component1, component2, component3, component4, init_cfg)
                self.linear = CustomLinear()

        checkpoint_path = tmp_path / "test.pth"
        torch.save(fxt_model.state_dict(), checkpoint_path)
        model = PratrainedModel(
            component1=FooConv1d(),
            component2=FooConv2d(),
            component3=FooLinear(),
            component4=FooLinearConv1d(linear=FooLinear(), conv1d=FooConv1d()),
            init_cfg={"type": "Pretrained", "checkpoint": checkpoint_path},
        )
        ori_layer_weight = model.linear.linear.weight.clone()
        ori_layer_bias = model.linear.linear.bias.clone()
        model.init_weights()

        assert (ori_layer_weight != model.linear.linear.weight).any()
        assert (ori_layer_bias != model.linear.linear.bias).any()

        # Test submodule.init_weights will be skipped if `is_init` is set
        # to True in root model
        model = FooModel(
            component1=FooConv1d(),
            component2=FooConv2d(),
            component3=FooLinear(),
            component4=FooLinearConv1d(linear=FooLinear(), conv1d=FooConv1d()),
            init_cfg=[
                {"type": "Constant", "val": 1, "bias": 2, "layer": "Linear"},
                {"type": "Constant", "val": 3, "bias": 4, "layer": "Conv1d"},
                {"type": "Constant", "val": 5, "bias": 6, "layer": "Conv2d"},
            ],
        )
        for child in model.children():
            child.init_weights = Mock()
        model.is_init = True
        model.init_weights()
        for child in model.children():
            child.init_weights.assert_not_called()

        # Test submodule.init_weights will be skipped if submodule's `is_init`
        # is set to True
        model = FooModel(
            component1=FooConv1d(),
            component2=FooConv2d(),
            component3=FooLinear(),
            component4=FooLinearConv1d(linear=FooLinear(), conv1d=FooConv1d()),
            init_cfg=[
                {"type": "Constant", "val": 1, "bias": 2, "layer": "Linear"},
                {"type": "Constant", "val": 3, "bias": 4, "layer": "Conv1d"},
                {"type": "Constant", "val": 5, "bias": 6, "layer": "Conv2d"},
            ],
        )
        for child in model.children():
            child.init_weights = Mock()
        model.component1.is_init = True
        model.reg.is_init = True
        model.init_weights()
        model.component1.init_weights.assert_not_called()
        model.component2.init_weights.assert_called_once()
        model.component3.init_weights.assert_called_once()
        model.component4.init_weights.assert_called_once()
        model.reg.init_weights.assert_not_called()


class TestModuleList:
    def test_modulelist_weight_init(self):
        layers = [
            FooConv1d(init_cfg={"type": "Constant", "layer": "Conv1d", "val": 0.0, "bias": 1.0}),
            FooConv2d(init_cfg={"type": "Constant", "layer": "Conv2d", "val": 2.0, "bias": 3.0}),
        ]
        modellist = ModuleList(layers)
        modellist.init_weights()
        assert torch.equal(modellist[0].conv1d.weight, torch.full(modellist[0].conv1d.weight.shape, 0.0))
        assert torch.equal(modellist[0].conv1d.bias, torch.full(modellist[0].conv1d.bias.shape, 1.0))
        assert torch.equal(modellist[1].conv2d.weight, torch.full(modellist[1].conv2d.weight.shape, 2.0))
        assert torch.equal(modellist[1].conv2d.bias, torch.full(modellist[1].conv2d.bias.shape, 3.0))
        # inner init_cfg has higher priority
        layers = [
            FooConv1d(init_cfg={"type": "Constant", "layer": "Conv1d", "val": 0.0, "bias": 1.0}),
            FooConv2d(init_cfg={"type": "Constant", "layer": "Conv2d", "val": 2.0, "bias": 3.0}),
        ]
        modellist = ModuleList(
            layers,
            init_cfg={"type": "Constant", "layer": ["Conv1d", "Conv2d"], "val": 4.0, "bias": 5.0},
        )
        modellist.init_weights()
        assert torch.equal(modellist[0].conv1d.weight, torch.full(modellist[0].conv1d.weight.shape, 0.0))
        assert torch.equal(modellist[0].conv1d.bias, torch.full(modellist[0].conv1d.bias.shape, 1.0))
        assert torch.equal(modellist[1].conv2d.weight, torch.full(modellist[1].conv2d.weight.shape, 2.0))
        assert torch.equal(modellist[1].conv2d.bias, torch.full(modellist[1].conv2d.bias.shape, 3.0))


class TestModuleDict:
    def test_moduledict_weight_init(self):
        layers = {
            "foo_conv_1d": FooConv1d(init_cfg={"type": "Constant", "layer": "Conv1d", "val": 0.0, "bias": 1.0}),
            "foo_conv_2d": FooConv2d(init_cfg={"type": "Constant", "layer": "Conv2d", "val": 2.0, "bias": 3.0}),
        }
        modeldict = ModuleDict(layers)
        modeldict.init_weights()
        assert torch.equal(
            modeldict["foo_conv_1d"].conv1d.weight,
            torch.full(modeldict["foo_conv_1d"].conv1d.weight.shape, 0.0),
        )
        assert torch.equal(
            modeldict["foo_conv_1d"].conv1d.bias,
            torch.full(modeldict["foo_conv_1d"].conv1d.bias.shape, 1.0),
        )
        assert torch.equal(
            modeldict["foo_conv_2d"].conv2d.weight,
            torch.full(modeldict["foo_conv_2d"].conv2d.weight.shape, 2.0),
        )
        assert torch.equal(
            modeldict["foo_conv_2d"].conv2d.bias,
            torch.full(modeldict["foo_conv_2d"].conv2d.bias.shape, 3.0),
        )
        # inner init_cfg has higher priority
        layers = {
            "foo_conv_1d": FooConv1d(init_cfg={"type": "Constant", "layer": "Conv1d", "val": 0.0, "bias": 1.0}),
            "foo_conv_2d": FooConv2d(init_cfg={"type": "Constant", "layer": "Conv2d", "val": 2.0, "bias": 3.0}),
        }
        modeldict = ModuleDict(
            layers,
            init_cfg={"type": "Constant", "layer": ["Conv1d", "Conv2d"], "val": 4.0, "bias": 5.0},
        )
        modeldict.init_weights()
        assert torch.equal(
            modeldict["foo_conv_1d"].conv1d.weight,
            torch.full(modeldict["foo_conv_1d"].conv1d.weight.shape, 0.0),
        )
        assert torch.equal(
            modeldict["foo_conv_1d"].conv1d.bias,
            torch.full(modeldict["foo_conv_1d"].conv1d.bias.shape, 1.0),
        )
        assert torch.equal(
            modeldict["foo_conv_2d"].conv2d.weight,
            torch.full(modeldict["foo_conv_2d"].conv2d.weight.shape, 2.0),
        )
        assert torch.equal(
            modeldict["foo_conv_2d"].conv2d.bias,
            torch.full(modeldict["foo_conv_2d"].conv2d.bias.shape, 3.0),
        )


class TestSequential:
    def test_sequential_model_weight_init(self):
        layers = [
            FooConv1d(init_cfg={"type": "Constant", "layer": "Conv1d", "val": 0.0, "bias": 1.0}),
            FooConv2d(init_cfg={"type": "Constant", "layer": "Conv2d", "val": 2.0, "bias": 3.0}),
        ]
        seq_model = Sequential(*layers)
        seq_model.init_weights()
        assert torch.equal(seq_model[0].conv1d.weight, torch.full(seq_model[0].conv1d.weight.shape, 0.0))
        assert torch.equal(seq_model[0].conv1d.bias, torch.full(seq_model[0].conv1d.bias.shape, 1.0))
        assert torch.equal(seq_model[1].conv2d.weight, torch.full(seq_model[1].conv2d.weight.shape, 2.0))
        assert torch.equal(seq_model[1].conv2d.bias, torch.full(seq_model[1].conv2d.bias.shape, 3.0))
        # inner init_cfg has higher priority
        layers = [
            FooConv1d(init_cfg={"type": "Constant", "layer": "Conv1d", "val": 0.0, "bias": 1.0}),
            FooConv2d(init_cfg={"type": "Constant", "layer": "Conv2d", "val": 2.0, "bias": 3.0}),
        ]
        seq_model = Sequential(
            *layers,
            init_cfg={"type": "Constant", "layer": ["Conv1d", "Conv2d"], "val": 4.0, "bias": 5.0},
        )
        seq_model.init_weights()
        assert torch.equal(seq_model[0].conv1d.weight, torch.full(seq_model[0].conv1d.weight.shape, 0.0))
        assert torch.equal(seq_model[0].conv1d.bias, torch.full(seq_model[0].conv1d.bias.shape, 1.0))
        assert torch.equal(seq_model[1].conv2d.weight, torch.full(seq_model[1].conv2d.weight.shape, 2.0))
        assert torch.equal(seq_model[1].conv2d.bias, torch.full(seq_model[1].conv2d.bias.shape, 3.0))
