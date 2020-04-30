"""
 Copyright (c) 2019 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import pytest
import torch
from torch import nn

from nncf.layers import NNCFConv2d, NNCFLinear, NNCFConvTranspose2d
from nncf.module_operations import UpdateWeight
from nncf.sparsity.rb.layers import RBSparsifyingWeight
from nncf.sparsity.rb.loss import SparseLoss


class TestModel(nn.Module):
    def __init__(self, layer, sparsify, size=1):
        super().__init__()
        self.size = size
        self.layer = layer
        if sparsify is None:
            sparsifier = RBSparsifyingWeight(size=size)
        else:
            sparsifier = RBSparsifyingWeight(sparsify=sparsify, size=size)
        self.op_key = self.layer.register_pre_forward_operation(UpdateWeight(sparsifier))

    @property
    def sparsifier(self):
        return self.layer.get_pre_op(self.op_key).operand

    def forward(self, x):
        return self.layer(x)


def sparse_model(module, sparsify, size=1):
    layer = module(size, size, size)
    return TestModel(layer, sparsify, size)


@pytest.mark.parametrize('module',
                         [NNCFLinear, NNCFConv2d, NNCFConvTranspose2d])
class TestSparseModules:
    def test_create_loss__with_defaults(self, module):
        model = sparse_model(module, None)
        loss = SparseLoss([model.sparsifier])
        assert not loss.disabled
        assert loss.target_sparsity_rate == 0
        assert loss.p == 0.05

    @pytest.mark.parametrize(('mask_value', 'ref_loss'),
                             ((None, 1),
                              (0, 0),
                              (0.3, 1),
                              (-0.3, 0)), ids=('default', 'zero', 'positive', 'negative'))
    def test_can_forward_sparse_module__with_frozen_mask(self, module, mask_value, ref_loss):
        model = sparse_model(module, False)
        sm = model.layer
        sm.weight.data.fill_(1)
        sm.bias.data.fill_(0)
        sw = model.sparsifier
        if mask_value is not None:
            new_mask = torch.zeros_like(sw.mask)
            new_mask.fill_(mask_value)
            sw.mask = new_mask
        input_ = torch.ones([1, 1, 1, 1])
        assert model(input_).item() == ref_loss

    @pytest.mark.parametrize(('sparsify', 'raising'), ((None, True), (True, False), (False, True)),
                             ids=('default', 'sparsify', 'frozen'))
    def test_calc_loss(self, module, sparsify, raising):
        model = sparse_model(module, sparsify)
        sw = model.sparsifier
        assert sw.sparsify is (False if sparsify is None else sparsify)
        loss = SparseLoss([model.sparsifier])
        try:
            assert loss() == 0
        except ZeroDivisionError:
            pytest.fail("Division by zero")
        except AssertionError:
            if not raising:
                pytest.fail("Exception is not expected")

    @pytest.mark.parametrize('sparsify', (None, True, False), ids=('default', 'sparsify', 'frozen'))
    class TestWithSparsify:
        def test_can_freeze_mask(self, module, sparsify):
            model = sparse_model(module, sparsify)
            sw = model.sparsifier
            if sparsify is None:
                sparsify = False
            assert sw.sparsify is sparsify
            assert sw.mask.numel() == 1

        def test_disable_loss(self, module, sparsify):
            model = sparse_model(module, sparsify)
            sw = model.sparsifier
            assert sw.sparsify is (False if sparsify is None else sparsify)
            loss = SparseLoss([model.sparsifier])
            loss.disable()
            assert not sw.sparsify

    @pytest.mark.parametrize(('target', 'expected_rate'),
                             ((None, 0),
                              (0, 1),
                              (0.5, 0.5),
                              (1, 0),
                              (1.5, None),
                              (-0.5, None)),
                             ids=('default', 'min', 'middle', 'max', 'more_than_max', 'less_then_min'))
    def test_get_target_sparsity_rate(self, module, target, expected_rate):
        model = sparse_model(module, None)
        loss = SparseLoss([model.sparsifier])
        if target is not None:
            loss.target = target
        actual_rate = None
        try:
            actual_rate = loss.target_sparsity_rate
            if expected_rate is None:
                pytest.fail("Exception should be raised")
        except IndexError:
            if expected_rate is not None:
                pytest.fail("Exception is not expected")
        if expected_rate is not None:
            assert actual_rate == expected_rate
