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

import torch.nn as nn
import torch.nn.functional as F

from nncf.algo_selector import create_dummy_forward_fn
from nncf.dynamic_graph import patch_torch_operators, reset_context
from nncf.layers import NNCFConv2d
from nncf.operations import UpdateWeight, UpdateInputs
from nncf.quantization import QuantizedNetwork
from nncf.quantization import SymmetricQuantizer
from nncf.quantization.layers import QuantizerConfig, QuantizationParams

patch_torch_operators()


def create_quantize_module(_, is_weights=False):
    params = QuantizationParams(signed=is_weights)
    return SymmetricQuantizer(QuantizerConfig(params, is_weights=is_weights))


def test_ambiguous_function():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Conv2d(1, 1, 1),
                nn.Conv2d(1, 1, 1)
            ])

        def forward(self, x):
            for layer in self.layers:
                x = F.relu(layer(x))

    reset_context('orig')
    reset_context('quantized_graphs')
    mod = Model()
    QuantizedNetwork(mod, create_quantize_module,
                     inputs_shape=(1, 1, 1, 1),
                     dummy_forward_fn=create_dummy_forward_fn((1, 1, 1, 1)))


def test_quantize_has_proper_is_weights_flag():
    class Model(nn.Module):
        def __init__(self, size=1):
            super().__init__()
            self.size = size
            self.conv = nn.Conv2d(size, size, size)

        def forward(self, x):
            return self.conv(x)

    model = Model()
    reset_context('orig')
    reset_context('quantized_graphs')

    quant_model = QuantizedNetwork(model, create_quantize_module, inputs_shape=(1, 1, 2, 2),
                                   dummy_forward_fn=create_dummy_forward_fn((1, 1, 2, 2)))
    for module in quant_model.modules():
        if isinstance(module, NNCFConv2d):
            for op in module.pre_ops.values():
                assert isinstance(op, (UpdateWeight, UpdateInputs))
                assert op.operand.is_weights is isinstance(op, UpdateWeight)
    for _, aq in quant_model.activation_quantizers.items():
        assert aq.is_weights is False
