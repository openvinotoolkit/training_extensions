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

from copy import copy
from collections import OrderedDict

from nncf.dynamic_graph.graph_builder import ModelInputInfo
from nncf.initialization import DataLoaderInitializeRunner, INITIALIZABLE_MODULES
from .layers import QUANTIZATION_MODULES, QuantizationMode, QuantizerConfig
from nncf.utils import get_all_modules_by_type
from .quantized_network import QuantizedNetwork
from ..algo_selector import COMPRESSION_ALGORITHMS
from ..compression_method_api import CompressionAlgorithm
from ..dynamic_graph.transform_graph import in_scope_list
from ..utils import get_module_by_node_name


@COMPRESSION_ALGORITHMS.register('quantization')
class Quantization(CompressionAlgorithm):
    def __init__(self, model, config,
                 input_infos: ModelInputInfo = None,
                 dummy_forward_fn=None):
        super().__init__(model, config, dummy_forward_fn)
        self.quantize_inputs = self.config.get('quantize_inputs', True)
        self.ignored_scopes = self.config.get('ignored_scopes')
        self.target_scopes = self.config.get('target_scopes')
        self.quantize_outputs = self.config.get('quantize_outputs', False)
        self.scopes_without_shape_matching = self.config.get('scopes_without_shape_matching', None)
        self.disable_function_quantization_hooks = self.config.get('disable_function_quantization_hooks', False)
        self.input_infos = input_infos

        params = self.config.get('activations', {})
        per_channel = params.get('per_channel', False)
        self.global_activations_params = QuantizerConfig(
            bits=params.get('bits', 8),
            mode=params.get('mode', QuantizationMode.SYMMETRIC),
            signedness_to_force=params.get('signed', None),
            per_channel=per_channel)

        params = self.config.get('weights', {})
        per_channel = params.get('per_channel', False)
        self.global_weights_params = QuantizerConfig(
            bits=params.get('bits', 8),
            mode=params.get('mode', QuantizationMode.SYMMETRIC),
            per_channel=per_channel)

        quantizable_subgraph_patterns = self.config.get('quantizable_subgraph_patterns', None)

        self._model = QuantizedNetwork(model, self.__create_quantize_module,
                                       input_infos=self.input_infos,
                                       ignored_scopes=self.ignored_scopes,
                                       target_scopes=self.target_scopes,
                                       quantize_inputs=self.quantize_inputs,
                                       quantize_outputs=self.quantize_outputs,
                                       dummy_forward_fn=dummy_forward_fn,
                                       quantizable_subgraph_patterns=quantizable_subgraph_patterns,
                                       scopes_without_shape_matching=self.scopes_without_shape_matching,
                                       disable_function_quantization_hooks=self.disable_function_quantization_hooks)
        self.is_distributed = False

    def distributed(self):
        self.is_distributed = True

    def initialize(self, data_loader=None):
        init_config = self.config.get('initializer', None)
        if init_config is None:
            return
        num_init_steps = init_config.get('num_init_steps', 1)
        if num_init_steps > 0:
            global_init_type = init_config.get('type', 'min_max')

            modules_to_init = OrderedDict()
            scope_overrides = self.config.get("scope_overrides", {})

            for module_type, _ in INITIALIZABLE_MODULES.registry_dict.items():
                module_dict = get_all_modules_by_type(self.model, module_type)
                for name, module in module_dict.items():
                    init_type = global_init_type
                    for overridden_scope in scope_overrides.keys():
                        if in_scope_list(name, overridden_scope):
                            init_config = scope_overrides[overridden_scope].get('initializer', {})
                            init_type = init_config.get("type", global_init_type)
                    modules_to_init[name] = (module, init_type)

            # NOTE: Order of modules must be the same to correctly broadcast parameters (e.g. input_low
            # and input_range)
            modules_to_init = OrderedDict(sorted(modules_to_init.items()))

            runner = DataLoaderInitializeRunner(self.model, modules_to_init)
            if self.is_distributed:
                # Multi-process data loading heavily slows down collecting statistics. The best option, when data
                # fetching is done in the same process a DataLoader is initialized, i.e. num_workers should be 0.
                num_workers = data_loader.num_workers
                data_loader.num_workers = 0

                runner.run(data_loader, num_init_steps, self.is_distributed)
                data_loader.num_workers = num_workers
            else:
                runner.run(data_loader, num_init_steps, self.is_distributed)
            self.model.rebuild_graph()

    def __create_quantize_module(self, parent_module_name, is_weights=False, input_shape=None):
        config = copy(self.global_weights_params) if is_weights else copy(self.global_activations_params)
        config.is_weights = is_weights

        scope_overrides = self.config.get("scope_overrides", {})
        for overridden_scope in scope_overrides.keys():
            if in_scope_list(parent_module_name, overridden_scope):
                config_overrides = scope_overrides[overridden_scope]
                if config_overrides.get("bits") is not None:
                    config.bits = config_overrides["bits"]
                if config_overrides.get("mode") is not None:
                    config.mode = config_overrides["mode"]
                if config_overrides.get("per_channel") is not None:
                    config.per_channel = config_overrides["per_channel"]
                if config_overrides.get("signed") is not None:
                    config.signedness_to_force = config_overrides["signed"]

        quantizer_cls = QUANTIZATION_MODULES.get(config.mode)
        if config.per_channel:
            if is_weights:
                module = get_module_by_node_name(self._model, parent_module_name)
                config.input_shape = module.weight.shape
            elif input_shape is not None:
                config.input_shape = input_shape
            else:
                raise RuntimeError("Unable to use per channel quantization for module {} activations -"
                                   " input shape is unknown".format(parent_module_name))

        return quantizer_cls(config)
