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

from .initializers import QUANTIZATION_INITIALIZERS
from .quantized_network import QuantizedNetwork
from ..algo_selector import COMPRESSION_ALGORITHMS
from ..compression_method_api import CompressionAlgorithm
from ..utils import get_all_modules_by_type


@COMPRESSION_ALGORITHMS.register('quantization')
class Quantization(CompressionAlgorithm):
    def __init__(self, model, config, input_size):
        super().__init__(model, config, input_size)

        self.bits = self.config.get('bits', 8)
        self.signed_activations = self.config.get('signed_activations', False)
        self.symmetric = self.config.get('symmetric', True)
        self.ignored_scopes = self.config.get('ignored_scopes', [])
        self.signed_activation_scopes = self.config.get('signed_activation_scopes', [])
        self.quantize_inputs = self.config.get('quantize_inputs', False)

        self._model = QuantizedNetwork(model, inputs_shape=self.input_size, bits=self.bits,
                                       activation_signed=self.signed_activations,
                                       symmetric=self.symmetric, ignored_scopes=self.ignored_scopes,
                                       signed_activation_scope=self.signed_activation_scopes,
                                       quantize_inputs=self.quantize_inputs)
        self.is_distributed = False

    def export_model(self, filename):
        self._model.export(filename)

    def distributed(self):
        self.is_distributed = True

    def initialize(self, data_loader=None):
        init_config = self.config.get('initializer', None)
        if init_config is not None:
            num_init_steps = init_config.get('num_init_steps', 1)
            if num_init_steps > 0:
                modules_to_init = get_all_modules_by_type(self.model, "Quantize")
                for module in modules_to_init.values():
                    module.init_stage = True
                init_type = init_config.get('type', 'min_max')
                initializer_cls = QUANTIZATION_INITIALIZERS.get(init_type)
                initializer = initializer_cls(modules_to_init, num_init_steps)
                initializer.run(self.model, data_loader, self.is_distributed)
                for module in modules_to_init.values():
                    module.init_stage = False
