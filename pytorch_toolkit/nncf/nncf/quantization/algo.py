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
from .layers import QUANTIZATION_MODULES, QuantizationMode, QuantizationParams, QuantizerConfig
from .quantized_network import QuantizedNetwork
from ..algo_selector import COMPRESSION_ALGORITHMS
from ..compression_method_api import CompressionAlgorithm
from ..dynamic_graph.transform_graph import in_scope_list
from ..utils import get_module_by_node_name


@COMPRESSION_ALGORITHMS.register('quantization')
class Quantization(CompressionAlgorithm):
    def __init__(self, model, config, input_size,
                 dummy_forward_fn=None):
        super().__init__(model, config, input_size)
        self.quantize_inputs = self.config.get('quantize_inputs', True)
        self.ignored_scopes = self.config.get('ignored_scopes')
        self.target_scopes = self.config.get('target_scopes')
        self.quantize_outputs = self.config.get('quantize_outputs', False)

        per_channel = self.config.get('per_channel', False)
        quantizable_subgraph_patterns = self.config.get('quantizable_subgraph_patterns', None)

        params = self.config.get('activations', {})
        self.activations_params = QuantizationParams(
            params.get('bits', 8),
            params.get('mode', QuantizationMode.SYMMETRIC),
            params.get('signed', False),
            params.get('signed_scope', []))
        self.params = self.config.get('weights', {})
        self.weights_params = QuantizationParams(
            params.get('bits', 8),
            params.get('mode', QuantizationMode.SYMMETRIC),
            per_channel=per_channel)
        self._model = QuantizedNetwork(model, self.__create_quantize_module,
                                       inputs_shape=self.input_size,
                                       ignored_scopes=self.ignored_scopes,
                                       quantize_inputs=self.quantize_inputs,
                                       quantize_outputs=self.quantize_outputs,
                                       dummy_forward_fn=dummy_forward_fn,
                                       quantizable_subgraph_patterns=quantizable_subgraph_patterns)
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
                init_type = init_config.get('type', 'min_max')
                initializer_cls = QUANTIZATION_INITIALIZERS.get(init_type)
                initializer = initializer_cls(self.model, num_init_steps)
                if self.is_distributed:
                    # Multi-process data loading heavily slows down collecting statistics. The best option, when data
                    # fetching is done in the same process a DataLoader is initialized, i.e. num_workers should be 0.
                    num_workers = data_loader.num_workers
                    data_loader.num_workers = 0
                    initializer.run(data_loader, self.is_distributed)
                    data_loader.num_workers = num_workers
                else:
                    initializer.run(data_loader, self.is_distributed)

    def __create_quantize_module(self, parent_module_name, is_weights=False):
        params = self.weights_params if is_weights else self.activations_params
        config = QuantizerConfig(params)
        config.is_weights = is_weights
        quantizer_cls = QUANTIZATION_MODULES.get(params.mode)
        if is_weights and params.per_channel:
            module = get_module_by_node_name(self._model, parent_module_name)
            config.input_shape = module.weight.shape
            config.per_channel = True
        within_signed_scope = in_scope_list('/'.join(parent_module_name), params.signed_scope)
        config.within_signed_scope = within_signed_scope
        return quantizer_cls(config)
