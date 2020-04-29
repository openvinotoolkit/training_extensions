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

import os

import pytest
import torch

from nncf.checkpoint_loading import load_state
from nncf.utils import get_all_modules_by_type

from examples.common.model_loader import load_model
from tests.quantization.test_functions import check_equal
from tests.test_helpers import BasicConvTestModel


@pytest.mark.parametrize(('model_name', 'ceil_mode'),
                         (('squeezenet1_1', True), ('squeezenet1_1_custom', False)), ids=('orig', 'custom'))
def test_load_sq_11_ceil_mode_is_ok(model_name, ceil_mode):
    model = load_model(model_name, pretrained=False)
    layers = get_all_modules_by_type(model, 'MaxPool2d')
    for _, pool_layer in layers.items():
        assert pool_layer.ceil_mode == ceil_mode


def test_export_sq_11_custom_is_ok(tmp_path):
    test_path = str(tmp_path.joinpath("test.onnx"))
    model = load_model('squeezenet1_1_custom', pretrained=False)
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy_input, test_path, verbose=False)
    os.remove(test_path)


def test_load_state_skips_not_matched_params__from_larger_to_smaller():
    ref_weights = BasicConvTestModel.default_weight()
    ref_bias = BasicConvTestModel.default_bias()
    model_save = BasicConvTestModel(out_channels=1, weight_init=2, bias_init=2)
    model_load = BasicConvTestModel(out_channels=2)

    num_loaded = load_state(model_load, model_save.state_dict())

    act_bias = model_load.conv.bias.data
    act_weights = model_load.conv.weight.data
    assert num_loaded == 0
    check_equal(act_bias, ref_bias)
    check_equal(act_weights, ref_weights)


def test_load_state_skips_not_matched_params__from_smaller_to_larger():
    ref_weights = torch.tensor([[[[3, 2],
                                  [2, 3]]]])
    ref_bias = torch.tensor([2.])
    model_save = BasicConvTestModel(out_channels=2)
    model_load = BasicConvTestModel(out_channels=1, weight_init=2, bias_init=2)

    num_loaded = load_state(model_load, model_save.state_dict())

    assert num_loaded == 0
    act_bias = model_load.conv.bias.data
    act_weights = model_load.conv.weight.data
    check_equal(act_bias, ref_bias)
    check_equal(act_weights, ref_weights)
