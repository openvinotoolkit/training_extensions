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

from nncf.layers import NNCFConv2d, NNCFLinear
from nncf.module_operations import UpdateWeight
from nncf.sparsity.layers import BinaryMask
from nncf.sparsity.magnitude.functions import normed_magnitude, abs_magnitude, calc_magnitude_binary_mask
from tests.test_helpers import fill_conv_weight, fill_linear_weight, fill_bias


class TestModel(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        sparsifier = BinaryMask(layer.weight.size())
        self.op_key = self.layer.register_pre_forward_operation(UpdateWeight(sparsifier))

    @property
    def sparsifier(self):
        return self.layer.get_pre_op(self.op_key).operand

    def forward(self, x):
        return self.layer(x)


@pytest.mark.parametrize(
    ('weight_importance', 'threshold', 'ref_output'),
    (
        (None, None, 38),
        (normed_magnitude, 10, 0),
        (normed_magnitude, 9, 0),
        (normed_magnitude, 0.5, 20),
        (normed_magnitude, 0.4, 38),
        (abs_magnitude, 10, 0),
        (abs_magnitude, 9, 20),
        (abs_magnitude, 0.5, 38),
        (abs_magnitude, 0.4, 38)
    )
)
def test_can_infer_magnitude_sparse_conv(weight_importance, threshold, ref_output):
    nncf_module = NNCFConv2d(1, 1, 2)
    sparse_model = TestModel(nncf_module)
    sparsifier = sparse_model.sparsifier
    fill_conv_weight(nncf_module, 9)
    fill_bias(nncf_module, 0)

    if threshold is not None:
        sparsifier.binary_mask = calc_magnitude_binary_mask(sparse_model.layer.weight,
                                                            weight_importance,
                                                            threshold)

    act_output = sparse_model(torch.ones([1, 1, 2, 2]))
    assert act_output.item() == ref_output


@pytest.mark.parametrize(
    ('weight_importance', 'threshold', 'ref_output'),
    (
        (None, None, 37),
        (normed_magnitude, 10, 0),
        (normed_magnitude, 9, 0),
        (normed_magnitude, 0.5, 10),
        (normed_magnitude, 0.4, 37),
        (abs_magnitude, 10, 0),
        (abs_magnitude, 9, 10),
        (abs_magnitude, 0.5, 37),
        (abs_magnitude, 0.4, 37)
    )
)
def test_can_infer_magnitude_sparse_linear(weight_importance, threshold, ref_output):
    nncf_module = NNCFLinear(4, 1)
    sparse_model = TestModel(nncf_module)
    sparsifier = sparse_model.sparsifier
    fill_linear_weight(nncf_module, 9)
    fill_bias(nncf_module, 0)

    if threshold is not None:
        sparsifier.binary_mask = calc_magnitude_binary_mask(sparse_model.layer.weight,
                                                            weight_importance,
                                                            threshold)

    act_output = sparse_model(torch.ones([1, 4]))
    assert act_output.item() == ref_output
