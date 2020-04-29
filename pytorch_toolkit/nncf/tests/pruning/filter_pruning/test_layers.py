"""
 Copyright (c) 2020 Intel Corporation
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

from nncf.layers import NNCFConv2d
from nncf.module_operations import UpdateWeight
from nncf.pruning.filter_pruning.layers import FilterPruningBlock, inplace_apply_filter_binary_mask, \
    apply_filter_binary_mask
from tests.test_helpers import fill_conv_weight, fill_bias


class TestFilterPruningBlockModel(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        pruning_op = FilterPruningBlock(layer.weight.size(0))
        self.op_key = self.layer.register_pre_forward_operation(UpdateWeight(pruning_op))

    @property
    def pruning_op(self):
        return self.layer.get_pre_op(self.op_key).operand

    def forward(self, x):
        return self.layer(x)


@pytest.mark.parametrize(
    ('weights_val', 'bias_val'),
    (
        (3, 0),
        (9, 0),
        (15, 1)
    )
)
def test_can_infer_magnitude_pruned_conv(weights_val, bias_val):
    """
    Check that NNCFConv2d with FilterPruningBlock as pre ops working exactly the same as
    normal nn.Conv2d.
    :param weights_val: value for filling weights
    :param bias_val: value for filling biases
    """
    nncf_module = NNCFConv2d(1, 1, 2)
    pytorch_module = nn.Conv2d(1, 1, 2)

    sparse_model = TestFilterPruningBlockModel(nncf_module)

    fill_conv_weight(nncf_module, weights_val)
    fill_bias(nncf_module, bias_val)

    fill_conv_weight(pytorch_module, weights_val)
    fill_bias(pytorch_module, bias_val)

    act_output = sparse_model(torch.ones([1, 1, 2, 2]))
    ref_output = pytorch_module(torch.ones([1, 1, 2, 2]))
    assert act_output.item() == ref_output


def test_assert_broadcastable_mask_and_weight_shape():
    nncf_module = NNCFConv2d(1, 2, 2)
    fill_conv_weight(nncf_module, 1)
    fill_bias(nncf_module, 1)

    mask = torch.zeros(10)

    with pytest.raises(RuntimeError):
        inplace_apply_filter_binary_mask(mask, nncf_module.weight.data)

    with pytest.raises(RuntimeError):
        apply_filter_binary_mask(mask, nncf_module.weight.data)


@pytest.mark.parametrize(('mask', 'reference_weight', 'reference_bias'),
                         [(torch.zeros(2), torch.zeros((2, 1, 2, 2)), torch.zeros(2)),
                          (torch.ones(2), torch.ones((2, 1, 2, 2)) + torch.eye(2), torch.ones(2)),
                          (torch.tensor([0, 1], dtype=torch.float32),
                           torch.cat([torch.zeros((1, 1, 2, 2)), torch.ones((1, 1, 2, 2)) + torch.eye(2)]),
                           torch.tensor([0, 1], dtype=torch.float32)),
                          ])
class TestApplyMasks:
    @staticmethod
    def test_inplace_apply_filter_binary_mask(mask, reference_weight, reference_bias):
        """
        Test that inplace_apply_filter_binary_mask changes the input weight and returns valid result.
        """
        nncf_module = NNCFConv2d(1, 2, 2)
        fill_conv_weight(nncf_module, 1)
        fill_bias(nncf_module, 1)

        result_weight = inplace_apply_filter_binary_mask(mask, nncf_module.weight.data)
        assert torch.allclose(result_weight, reference_weight)
        assert torch.allclose(nncf_module.weight, reference_weight)

        result_bias = inplace_apply_filter_binary_mask(mask, nncf_module.bias.data)
        assert torch.allclose(result_bias, reference_bias)
        assert torch.allclose(nncf_module.bias, reference_bias)

    @staticmethod
    def test_apply_filter_binary_mask(mask, reference_weight, reference_bias):
        """
        Test that apply_filter_binary_mask not changes the input weight and returns valid result.
        """
        nncf_module = NNCFConv2d(1, 2, 2)
        fill_conv_weight(nncf_module, 1)
        fill_bias(nncf_module, 1)

        original_weight = nncf_module.weight.data.detach().clone()
        original_bias = nncf_module.bias.data.detach().clone()

        result = apply_filter_binary_mask(mask, nncf_module.weight.data)
        assert torch.allclose(nncf_module.weight, original_weight)
        assert torch.allclose(result, reference_weight)

        result_bias = apply_filter_binary_mask(mask, nncf_module.bias.data)
        assert torch.allclose(result_bias, reference_bias)
        assert torch.allclose(nncf_module.bias, original_bias)
