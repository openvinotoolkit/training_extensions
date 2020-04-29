"""
 Copyright (c) 2019-2020 Intel Corporation
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

from collections import OrderedDict
from enum import Enum
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import functools
import networkx as nx
import numpy as np
import operator
import shutil
import torch
from texttable import Texttable
from torch import nn

from nncf.algo_selector import COMPRESSION_ALGORITHMS
from nncf.compression_method_api import CompressionAlgorithmBuilder, CompressionAlgorithmController
from nncf.config import Config
from nncf.debug import is_debug, DebugInterface, CallCountTracker
from nncf.dynamic_graph.context import OperatorInput, TracingContext, \
    InputAgnosticOperationExecutionContext, Scope
from nncf.dynamic_graph.function_input_quantization import FUNCTIONS_TO_QUANTIZE
from nncf.dynamic_graph.graph import NNCFNode
from nncf.dynamic_graph.graph import NNCFNodeExpression as N, NNCFGraph
from nncf.dynamic_graph.patch_pytorch import get_arg_positions_to_quantize
from nncf.dynamic_graph.transform_graph import is_nncf_module
from nncf.hw_config import HWConfig
from nncf.initialization import DataLoaderInitializeRunner
from nncf.module_operations import UpdateWeight, UpdateInputs
from nncf.nncf_logger import logger as nncf_logger
from nncf.nncf_network import NNCFNetwork, CompressionModuleType, InsertionInfo, InsertionCommand, OperationPriority, \
    InsertionPoint, InsertionType, InsertionPointGraph, InsertionPointGraphNodeType
from nncf.quantization.init_precision import PrecisionInitializerFactory
from nncf.quantization.layers import QUANTIZATION_MODULES, QuantizationMode, QuantizerConfig, BaseQuantizer
from nncf.quantization.quantizer_propagation import QuantizerPropagationSolver, QuantizerPropagationStateGraph
from nncf.utils import get_all_modules_by_type, in_scope_list, is_main_process
from nncf.utils import get_state_dict_names_with_modules


class QuantizerSetupType(Enum):
    PATTERN_BASED = "pattern_based"
    PROPAGATION_BASED = "propagation_based"


class QuantizationConstraints:
    REF_QCONF_OBJ = QuantizerConfig()

    def __init__(self, **kwargs):
        """Use attribute names of QuantizerConfig as arguments
        to set up constraints.
        E.g. QuantizationConstraint(bits=8, per_channel=True) will set up
        a constraint that corresponds to all 8-bit per-channel quantizers, either
        symmetric or asymmetric, either signed or unsigned."""

        for attr_name in kwargs:
            if not hasattr(QuantizationConstraints.REF_QCONF_OBJ, attr_name):
                raise RuntimeError("Invalid constraint - QuantizerConfig has no attribute '{}'".format(attr_name))
        self.qconf_attr_vs_constraint_dict = kwargs

    def apply_constraints_to(self, qconfig: QuantizerConfig) -> QuantizerConfig:
        for attr_name, constraint in self.qconf_attr_vs_constraint_dict.items():
            if constraint is not None:
                setattr(qconfig, attr_name, constraint)
        return qconfig

    def is_config_compatible(self, qconfig: QuantizerConfig) -> bool:
        is_compatible = True
        for attr_name, constraint in self.qconf_attr_vs_constraint_dict.items():
            if constraint is not None:
                qconf_attr_value = getattr(qconfig, attr_name)
                if qconf_attr_value != constraint:
                    is_compatible = False
        return is_compatible


class QuantizerGroup(Enum):
    ACTIVATIONS = "activations"
    WEIGHTS = "weights"


@COMPRESSION_ALGORITHMS.register('quantization')
class QuantizationBuilder(CompressionAlgorithmBuilder):
    def __init__(self, config):
        super().__init__(config)

        self.quantize_inputs = self.config.get('quantize_inputs', True)
        self.quantize_outputs = self.config.get('quantize_outputs', False)
        self.disable_function_quantization_hooks = self.config.get('disable_function_quantization_hooks', False)

        self._debug_interface = QuantizationDebugInterface() if is_debug() else None

        self._quantized_weight_modules_registry = OrderedDict()
        self._quantized_inputs_modules_registry = OrderedDict()
        self._weight_quantizers = OrderedDict()  # Quantizers applied via UpdateWeights
        self._non_weight_quantizers = OrderedDict()  # All the other quantizers
        self._processed_function_quantizers = set()
        self._processed_input_agnostic_op_exec_contexts = set()

        self.global_quantizer_contraints = {}  # type: Dict[QuantizerGroup, QuantizationConstraints]
        self._ignored_scopes_per_group = {}  # type: Dict[QuantizerGroup, List[str]]
        self._target_scopes_per_group = {}  # type: Dict[QuantizerGroup, List[str]]

        for quantizer_group in QuantizerGroup:
            self._parse_group_params(self.config, quantizer_group)

        self.quantizer_setup_type = QuantizerSetupType.PATTERN_BASED  # TODO: determine from config
        self.quantizable_subgraph_patterns = self.config.get('quantizable_subgraph_patterns', None)
        self.hw_config = None
        hw_config_type = self.config.get("hw_config_type")
        if hw_config_type is not None:
            hw_config_path = HWConfig.get_path_to_hw_config(hw_config_type)
            self.hw_config = HWConfig.from_json(hw_config_path)
            self.quantizer_setup_type = QuantizerSetupType.PROPAGATION_BASED

    def _parse_group_params(self, quant_config: Config, quantizer_group: QuantizerGroup):
        group_name = quantizer_group.value
        params_dict = quant_config.get(group_name, {})
        self.global_quantizer_contraints[quantizer_group] = QuantizationConstraints(
            bits=params_dict.get('bits'),
            mode=params_dict.get('mode'),
            signedness_to_force=params_dict.get('signed'),
            per_channel=params_dict.get('per_channel')
        )
        self._ignored_scopes_per_group[quantizer_group] = params_dict.get('ignored_scopes')
        self._target_scopes_per_group[quantizer_group] = params_dict.get('target_scopes')

    def apply_to(self, target_model: NNCFNetwork) -> NNCFNetwork:
        insertion_commands = self._quantize_weights(target_model) + self._quantize_activations(target_model)
        if self.quantize_inputs:
            insertion_commands += self._quantize_inputs(target_model, insertion_commands)

        quantization_types = [class_type.__name__ for class_type in QUANTIZATION_MODULES.registry_dict.values()]

        # At this point the NNCF module quantization modules are not in the target_model yet,
        # therefore it is extended with the corresponding registries tracked during weights/inputs quantizations
        self._all_quantizations = get_state_dict_names_with_modules(target_model, quantization_types)
        self._all_quantizations.update(self._quantized_weight_modules_registry)
        self._all_quantizations.update(self._quantized_inputs_modules_registry)

        for command in insertion_commands:
            target_model.register_insertion_command(command)

        target_model.register_algorithm(self)

        if self._debug_interface is not None:
            target_model.debug_interface.add_interface(self._debug_interface)
        return target_model

    def build_controller(self, target_model: NNCFNetwork) -> CompressionAlgorithmController:
        return QuantizationController(target_model,
                                      self.config,
                                      self._debug_interface,
                                      self._quantized_weight_modules_registry,
                                      self._quantized_inputs_modules_registry,
                                      self._weight_quantizers,
                                      self._non_weight_quantizers)

    def __get_default_qconfig(self, constraints: QuantizationConstraints = None):
        qconfig = QuantizerConfig(bits=8,
                                  mode=QuantizationMode.SYMMETRIC,
                                  signedness_to_force=None,
                                  per_channel=False)
        if constraints is not None:
            qconfig = constraints.apply_constraints_to(qconfig)
        return qconfig

    def __get_scoped_quantizer_config(self, target_model: NNCFNetwork,
                                      parent_module_scope_str: str, is_weights=False, input_shape=None):
        group = QuantizerGroup.WEIGHTS if is_weights else QuantizerGroup.ACTIVATIONS
        qconfig = self.__get_default_qconfig(constraints=self.global_quantizer_contraints[group])
        qconfig.is_weights = is_weights

        scope_overrides = self.config.get("scope_overrides", {})
        for overridden_scope in scope_overrides.keys():
            if in_scope_list(parent_module_scope_str, overridden_scope):
                config_overrides = scope_overrides[overridden_scope]
                if config_overrides.get("bits") is not None:
                    qconfig.bits = config_overrides["bits"]
                if config_overrides.get("mode") is not None:
                    qconfig.mode = config_overrides["mode"]
                if config_overrides.get("per_channel") is not None:
                    qconfig.per_channel = config_overrides["per_channel"]
                if config_overrides.get("signed") is not None:
                    qconfig.signedness_to_force = config_overrides["signed"]
        if qconfig.per_channel:
            if is_weights:
                module = target_model.get_module_by_scope(Scope.from_str(parent_module_scope_str))
                qconfig.input_shape = module.weight.shape
            elif input_shape is not None:
                qconfig.input_shape = input_shape
            else:
                raise RuntimeError("Unable to use per channel quantization for module {} activations -"
                                   " input shape is unknown".format(parent_module_scope_str))
        return qconfig

    def __create_quantize_module(self, qconfig: QuantizerConfig):
        quantizer_cls = QUANTIZATION_MODULES.get(qconfig.mode)
        return quantizer_cls(qconfig)

    def _make_quantizable_subgraph_pattern(self):
        full_pattern = self._make_default_quantizable_subgraph_pattern()
        if self.quantizable_subgraph_patterns is not None:
            for pattern in self.quantizable_subgraph_patterns:
                if not isinstance(pattern, str):
                    custom_pattern = functools.reduce(operator.add,
                                                      [N(node) for node in pattern])
                else:
                    custom_pattern = N(pattern)
                full_pattern = full_pattern | custom_pattern
        return full_pattern

    def _quantize_weights(self, target_model: NNCFNetwork) -> List[InsertionCommand]:
        device = next(target_model.parameters()).device
        modules = target_model.get_nncf_modules()
        insertion_commands = []
        if self.hw_config is not None:
            meta_vs_qconfig_map = self.hw_config.get_metatype_vs_quantizer_configs_map(for_weights=True)
        for module_scope, module in modules.items():
            if not self._should_consider_scope_for_group(str(module_scope), QuantizerGroup.WEIGHTS):
                nncf_logger.info("Ignored adding Weight quantizer in scope: {}".format(module_scope))
                continue

            self._quantized_weight_modules_registry[str(module_scope)] = module

            nncf_logger.info("Adding signed Weight quantizer in scope: {}".format(module_scope))
            if self.hw_config is None:
                qconfig = self.__get_scoped_quantizer_config(target_model, str(module_scope), is_weights=True)
            else:
                associated_ops = target_model.get_insertion_point_graph().get_op_nodes_in_scope(module_scope)
                if not associated_ops:
                    raise RuntimeError(
                        "Could not find a patched operation corresponding to NNCF module scope {}".format(
                            str(module_scope)))
                assert len(associated_ops) == 1, "NNCF module has more than 1 associated graph operation node - " \
                                                 "cannot make sure that weight quantization will be correct"
                graph_operation = associated_ops[0]
                metatype = graph_operation[InsertionPointGraph.OPERATOR_METATYPE_NODE_ATTR]
                qconfig_list = meta_vs_qconfig_map[metatype]

                try:
                    qconfig = self._select_final_qconfig(qconfig_list,
                                                         self.global_quantizer_contraints[QuantizerGroup.WEIGHTS])
                except RuntimeError:
                    err_msg = "Quantization parameter constraints specified in NNCF config are incompatible with HW "
                    err_msg += "capabilities as specified in HW config type '{}'. ".format(self.hw_config.target_device)
                    err_msg += "First conflicting quantizer location: {}".format(str(module_scope))
                    raise RuntimeError(err_msg)

                qconfig.input_shape = module.weight.shape

            quantizer = self.__create_quantize_module(qconfig)
            op = UpdateWeight(quantizer).to(device)
            # TODO: separate insertion point semantic for weights and activations
            insertion_commands.append(InsertionCommand(
                InsertionPoint(
                    InputAgnosticOperationExecutionContext("", module_scope, 0),
                    InsertionType.NNCF_MODULE_PRE_OP), op, OperationPriority.QUANTIZATION_PRIORITY))
            self._weight_quantizers[self.WeightQuantizerKey(module_scope)] = quantizer
        return insertion_commands

    class ActivationQuantizationHook:
        """Cannot simply register the quantizer module as a callable hook, since we need to call
        a thread-local version of the quantizer module during base module execution."""

        def __init__(self, context: TracingContext, ia_op_exec_context: InputAgnosticOperationExecutionContext,
                     debug_interface: 'QuantizationDebugInterface' = None):
            self.compressed_context = context
            self.ia_op_exec_context = ia_op_exec_context
            self.debug_interface = debug_interface

        def __call__(self, *args, **kwargs):
            if self.debug_interface is not None:
                self.debug_interface.register_activation_quantize_call(str(self.ia_op_exec_context))
            replica = self.compressed_context.base_module_thread_local_replica
            return replica.activation_quantizers[str(self.ia_op_exec_context)](*args, **kwargs)

    def _quantize_activations(self, target_model: NNCFNetwork) -> List[InsertionCommand]:
        target_model.register_compression_module_type(CompressionModuleType.ACTIVATION_QUANTIZER)

        if self.quantizer_setup_type == QuantizerSetupType.PATTERN_BASED:
            insertion_commands = self._quantize_post_pattern_activations(target_model)
        elif self.quantizer_setup_type == QuantizerSetupType.PROPAGATION_BASED:
            insertion_point_graph = target_model.get_insertion_point_graph()
            if self._debug_interface:
                self._debug_interface.visualize_insertion_point_graph(insertion_point_graph)
            prop_graph_solver = QuantizerPropagationSolver(debug_interface=self._debug_interface,
                                                           hw_config=self.hw_config)
            merged_ip_graph = insertion_point_graph.get_ip_graph_with_merged_hw_optimized_operations(self.hw_config)
            insertion_data = prop_graph_solver.run_on_ip_graph(merged_ip_graph)
            insertion_commands = []

            original_nncf_graph = target_model.get_original_graph()
            for insertion_info, quantizer_config_list in insertion_data.items():
                # Tailored for post-hook quantization and first output quantization only
                quantizer_input_shape = original_nncf_graph.get_output_shapes_for_ia_op_exec_context(
                    insertion_info.op_exec_context.input_agnostic)[0]

                try:
                    quantizer_config = self._select_final_qconfig(quantizer_config_list,
                                                                  self.global_quantizer_contraints[
                                                                      QuantizerGroup.ACTIVATIONS])
                except RuntimeError:
                    err_msg = "Quantization parameter constraints specified in NNCF config are incompatible with HW "
                    err_msg += "capabilities as specified in HW config type '{}'. ".format(self.hw_config.target_device)
                    err_msg += "First conflicting quantizer location: "
                    err_msg += str(insertion_info.op_exec_context.input_agnostic)
                    raise RuntimeError(err_msg)

                quantizer_config.input_shape = quantizer_input_shape
                insertion_commands.append(
                    self._quantize_single_activation(target_model, insertion_info, quantizer_config))
        else:
            raise RuntimeError("Invalid quantizer setup type!")

        if not self.disable_function_quantization_hooks:
            insertion_commands += self._quantize_free_function_inputs(target_model)
        return insertion_commands

    def _select_final_qconfig(self, quantizer_config_list: Optional[List[QuantizerConfig]],
                              constraints: QuantizationConstraints) -> QuantizerConfig:
        if quantizer_config_list is None:
            # TODO: This case corresponds to allowing to use any quantization configuration
            # supported by HW. Need to parse this from HW config instead of using a global
            # default config.
            return self.__get_default_qconfig()

        constrained_quantizer_config_list = list(filter(
            constraints.is_config_compatible,
            quantizer_config_list
        ))

        if not constrained_quantizer_config_list:
            raise RuntimeError()

        # Quantizer config list entries should arrive in the same order as they are listed
        # in the HW config, where they are sorted by descending order of priority
        return constrained_quantizer_config_list[0]

    def _quantize_post_pattern_activations(self, target_model: NNCFNetwork) -> List[InsertionCommand]:
        pattern = self._make_quantizable_subgraph_pattern()
        target_insertion_infos = target_model.get_post_pattern_insertion_points(pattern)
        insertion_commands = []

        for insertion_info in target_insertion_infos:
            ia_op_exec_context = insertion_info.op_exec_context.input_agnostic
            operator_scope_str = str(ia_op_exec_context)

            if not self.quantize_outputs and insertion_info.is_output:
                nncf_logger.info("Ignored adding Activation Quantize "
                                 "in scope (output scope, quantize_outputs=False): {}".format(operator_scope_str))
                continue
            if not self._should_consider_scope_for_group(operator_scope_str, QuantizerGroup.ACTIVATIONS):
                nncf_logger.info("Ignored adding Activation quantizer in scope: {}".format(operator_scope_str))
                continue

            qconfig = self.__get_scoped_quantizer_config(target_model, operator_scope_str,
                                                         is_weights=False,
                                                         input_shape=insertion_info.shape_to_operate_on)
            insertion_commands.append(self._quantize_single_activation(target_model, insertion_info, qconfig))

        # NOTE: Order of activations must be the same to correctly broadcast parameters (e.g. scales) in distributed
        # mode (see call of `_dist_broadcast_coalesced` in torch/nn/parallel/distributed.py for more details)
        # pylint: disable=protected-access
        target_model.sort_compression_modules(CompressionModuleType.ACTIVATION_QUANTIZER)
        return insertion_commands

    def _quantize_single_activation(self, target_model: NNCFNetwork,
                                    insertion_info: InsertionInfo,
                                    quantizer_config: QuantizerConfig) -> InsertionCommand:
        ia_op_exec_context = insertion_info.op_exec_context.input_agnostic
        quantizer_key = self.NonWeightQuantizerKey(ia_op_exec_context)
        operator_scope_str = str(quantizer_key.ia_op_exec_context)
        device = next(target_model.parameters()).device

        if ia_op_exec_context in self._processed_input_agnostic_op_exec_contexts:
            raise RuntimeError(
                "Ambiguous call to {fn} with call order {co} in current scope. "
                "Cannot insert quantization hooks "
                "automatically!".format(fn=ia_op_exec_context.operator_name, co=ia_op_exec_context.call_order)
            )
        self._processed_input_agnostic_op_exec_contexts.add(ia_op_exec_context)

        assert operator_scope_str not in target_model.get_compression_modules_by_type(
            CompressionModuleType.ACTIVATION_QUANTIZER)

        quantizer = self.__create_quantize_module(quantizer_config).to(device)
        target_model.add_compression_module(operator_scope_str, quantizer,
                                            CompressionModuleType.ACTIVATION_QUANTIZER)
        self._non_weight_quantizers[quantizer_key] = quantizer

        nncf_logger.info("Adding {} Activation Quantize in scope: {}".format(
            "signed" if quantizer.signed else
            "unsigned", operator_scope_str
        ))

        hook = self.ActivationQuantizationHook(target_model.get_tracing_context(),
                                               ia_op_exec_context,
                                               self._debug_interface)

        self._processed_input_agnostic_op_exec_contexts.add(ia_op_exec_context)
        return InsertionCommand(InsertionPoint(ia_op_exec_context,
                                               InsertionType.OPERATOR_POST_HOOK),
                                hook,
                                OperationPriority.QUANTIZATION_PRIORITY)

    def _quantize_inputs(self, target_model: NNCFNetwork,
                         prev_weight_and_activation_quantizer_insertion_commands: List[InsertionCommand]) -> \
            List[InsertionCommand]:
        device = next(target_model.parameters()).device
        graph_roots = target_model.get_original_graph().get_input_nodes()

        # Have to handle the situation when the input node of the network is an NNCF module -
        # to quantize inputs in this case we will have to use UpdateInputs module pre-op,

        # Traverse down starting from graph roots to search for the first node which belongs to a NNCF module
        # and has no UpdateInputs pre-op

        def traverse_function(node: NNCFNode, output) -> Tuple[bool, List[NNCFNode]]:
            module = target_model.get_module_by_scope(node.op_exec_context.scope_in_model)
            if is_nncf_module(module):
                current_node_scope = node.op_exec_context.scope_in_model
                module_op_insertion_commands = []
                for comm in prev_weight_and_activation_quantizer_insertion_commands:
                    if current_node_scope in comm.insertion_point.ia_op_exec_context.scope_in_model:
                        module_op_insertion_commands.append(comm)
                pre_op_insertion_commands = filter(
                    lambda comm: comm.insertion_point.insertion_type == InsertionType.NNCF_MODULE_PRE_OP,
                    module_op_insertion_commands)
                update_inputs_count = sum(1 for comm in pre_op_insertion_commands if isinstance(comm.fn, UpdateInputs))
                if update_inputs_count == 0:
                    output.append(node)
                    return True, output
            else:
                current_node_ia_op_exec_context = node.op_exec_context.input_agnostic
                op_hook_insertion_commands = []
                for comm in prev_weight_and_activation_quantizer_insertion_commands:
                    if current_node_ia_op_exec_context == comm.insertion_point.ia_op_exec_context:
                        op_hook_insertion_commands.append(comm)
                if op_hook_insertion_commands:
                    return True, output

            return False, output

        nncf_module_input_nodes = set()
        for node in graph_roots:
            scope_str = str(node.op_exec_context.scope_in_model)
            if self._should_consider_scope_for_group(scope_str, QuantizerGroup.ACTIVATIONS):
                nncf_module_input_nodes.update(
                    target_model.get_original_graph().traverse_graph(node, traverse_function))

        insertion_commands = []
        nncf_scope_module_dict = target_model.get_nncf_modules()
        for module_input_node in nncf_module_input_nodes:
            op_scope = module_input_node.op_exec_context.input_agnostic.scope_in_model
            module = None
            scope = None
            for nncf_scope, nncf_module in nncf_scope_module_dict.items():
                if op_scope in nncf_scope:
                    module = nncf_module
                    scope = nncf_scope
                    break

            self._quantized_inputs_modules_registry[str(scope)] = module

            # Only use the shape of the 0-th input info specified in config. TODO: fix this
            input_shape = target_model.input_infos[0].shape if target_model.input_infos is not None else None
            qconfig = self.__get_scoped_quantizer_config(target_model, str(scope), is_weights=False,
                                                         input_shape=input_shape)
            quantizer = self.__create_quantize_module(qconfig)

            nncf_logger.info("Adding {} NNCF module input quantizer in scope: {}".format(
                "signed" if quantizer.signed else "unsigned", str(scope)
            ))

            # TODO: separate insertion point semantic for weights and activations
            insertion_commands.append(
                InsertionCommand(InsertionPoint(InputAgnosticOperationExecutionContext("", scope, 0),
                                                InsertionType.NNCF_MODULE_PRE_OP),
                                 UpdateInputs(quantizer).to(device),
                                 OperationPriority.QUANTIZATION_PRIORITY))
            ia_op_exec_context = module_input_node.op_exec_context.input_agnostic
            self._non_weight_quantizers[self.InputQuantizerKey(ia_op_exec_context)] = quantizer

        return insertion_commands

    def _should_consider_scope_for_group(self, scope_str: str, group: QuantizerGroup) -> bool:
        if self.target_scopes is not None or self._target_scopes_per_group[group] is not None:
            if in_scope_list(scope_str, self.target_scopes):
                return True
            if in_scope_list(scope_str, self._target_scopes_per_group[group]):
                return True

            return False

        if in_scope_list(scope_str, self.ignored_scopes):
            return False
        if in_scope_list(scope_str, self._ignored_scopes_per_group[group]):
            return False

        return True

    class QuantizerKey:
        def get_base(self):
            raise NotImplementedError

        def get_suffix(self) -> str:
            raise NotImplementedError

        def __str__(self):
            return str(self.get_base()) + self.get_suffix()

        def __hash__(self):
            return hash((self.get_base(), self.get_suffix()))

    class WeightQuantizerKey(QuantizerKey):
        def __init__(self, scope: 'Scope'):
            self.scope = scope

        def get_base(self) -> 'Scope':
            return self.scope

        def get_suffix(self) -> str:
            return 'module_weight'

    class NonWeightQuantizerKey(QuantizerKey):
        def __init__(self, ia_op_exec_context: InputAgnosticOperationExecutionContext):
            self.ia_op_exec_context = ia_op_exec_context

        def get_base(self) -> 'InputAgnosticOperationExecutionContext':
            return self.ia_op_exec_context

        def get_suffix(self) -> str:
            return ''

    class InputQuantizerKey(NonWeightQuantizerKey):
        def get_base(self) -> 'Scope':
            return self.ia_op_exec_context.scope_in_model

        def get_suffix(self) -> str:
            return 'module_input'

    class FunctionQuantizerKey(NonWeightQuantizerKey):
        def __init__(self, ia_op_exec_context: InputAgnosticOperationExecutionContext, input_arg_idx: int):
            super().__init__(ia_op_exec_context)
            self.input_arg_idx = input_arg_idx

        def get_suffix(self) -> str:
            return "_input" + str(self.input_arg_idx)

    class FunctionQuantizationPreHook:
        """Cannot simply register the quantizer module as a callable hook, since we need to call
        a thread-local version of the quantizer module during base module execution."""

        def __init__(self, context: TracingContext, func_in_quant_info: 'FunctionQuantizerKey',
                     debug_interface: 'QuantizationDebugInterface' = None):
            self.compressed_context = context
            self.func_in_quant_info = func_in_quant_info
            self.debug_interface = debug_interface

        def __call__(self, op_inputs: OperatorInput):
            quantizer_dict_key = str(self.func_in_quant_info)
            if self.debug_interface is not None:
                self.debug_interface.register_function_quantizer_call(quantizer_dict_key)
            replica = self.compressed_context.base_module_thread_local_replica
            idx = self.func_in_quant_info.input_arg_idx
            op_inputs.op_args[idx] = replica.function_quantizers[quantizer_dict_key](op_inputs.op_args[idx])
            return op_inputs

    def _quantize_free_function_inputs(self, target_model: NNCFNetwork) -> List[InsertionCommand]:
        device = next(target_model.parameters()).device

        if not FUNCTIONS_TO_QUANTIZE:
            return []
        pattern = N(FUNCTIONS_TO_QUANTIZE[0].name)
        for i in range(1, len(FUNCTIONS_TO_QUANTIZE)):
            pattern |= N(FUNCTIONS_TO_QUANTIZE[i].name)

        target_insertion_infos = target_model.get_post_pattern_insertion_points(pattern,
                                                                                omit_nodes_in_nncf_modules=True)
        insertion_commands = []

        target_model.register_compression_module_type(CompressionModuleType.FUNCTION_QUANTIZER)
        for insertion_info in target_insertion_infos:
            ia_op_exec_context = insertion_info.op_exec_context.input_agnostic
            scope_str = str(ia_op_exec_context.scope_in_model)

            if not self._should_consider_scope_for_group(scope_str, QuantizerGroup.ACTIVATIONS):
                nncf_logger.info("Ignored adding function input quantizer in scope: {}".format(scope_str))
                continue

            function_arg_positions_to_quantize = get_arg_positions_to_quantize(ia_op_exec_context.operator_name)
            assert function_arg_positions_to_quantize is not None, "Function with inputs to be quantized has " \
                                                                   "no info struct registered in " \
                                                                   "QUANTIZED_INPUT_FUNCTIONS!"

            for input_arg_idx in function_arg_positions_to_quantize:
                ip_arg_quant_key = self.FunctionQuantizerKey(ia_op_exec_context, input_arg_idx)

                if ip_arg_quant_key in self._processed_function_quantizers:
                    raise RuntimeError(
                        "Ambiguous call to {fn} with call order {co} and argname {arg} in current scope. "
                        "Cannot insert quantization hooks "
                        "automatically!".format(fn=ia_op_exec_context.operator_name,
                                                co=ia_op_exec_context.call_order,
                                                arg=input_arg_idx)
                    )

                self._processed_function_quantizers.add(ip_arg_quant_key)

                ip_arg_quant_name = str(ip_arg_quant_key)
                assert ip_arg_quant_name not in target_model.get_compression_modules_by_type(
                    CompressionModuleType.FUNCTION_QUANTIZER)
                input_shape = insertion_info.op_exec_context.tensor_metas[0].shape

                qconfig = self.__get_scoped_quantizer_config(target_model, scope_str,
                                                             is_weights=False,
                                                             input_shape=input_shape)
                quantizer_module = self.__create_quantize_module(qconfig).to(device)
                target_model.add_compression_module(ip_arg_quant_name, quantizer_module,
                                                    CompressionModuleType.FUNCTION_QUANTIZER)

                nncf_logger.info("Adding {} Function Quantize: {}".format(
                    "signed" if quantizer_module.signed else
                    "unsigned", ip_arg_quant_name))

                hook = self.FunctionQuantizationPreHook(target_model.get_tracing_context(),
                                                        ip_arg_quant_key,
                                                        self._debug_interface)
                insertion_commands.append(InsertionCommand(InsertionPoint(ia_op_exec_context,
                                                                          InsertionType.OPERATOR_PRE_HOOK),
                                                           hook,
                                                           OperationPriority.QUANTIZATION_PRIORITY))
                self._non_weight_quantizers[ip_arg_quant_key] = quantizer_module
        # NOTE: Order of input quantizers must be the same to correctly broadcast parameters (e.g. scales) in
        # distributed mode (see call of `_dist_broadcast_coalesced` in torch/nn/parallel/distributed.py for more
        # details) pylint: disable=protected-access
        target_model.sort_compression_modules(CompressionModuleType.FUNCTION_QUANTIZER)
        return insertion_commands

    @staticmethod
    def _make_default_quantizable_subgraph_pattern():
        import nncf.dynamic_graph.patterns as p
        pattern = p.LINEAR_OPS | p.ARITHMETIC | p.ANY_BN_RELU_COMBO | \
                  p.LINEAR_OPS + p.ANY_BN_RELU_COMBO | p.ARITHMETIC + p.ANY_BN_RELU_COMBO | p.SINGLE_OPS | p.MATMUL
        return pattern


class QuantizationController(CompressionAlgorithmController):
    def __init__(self, target_model: NNCFNetwork,
                 quantization_config: Config,
                 debug_interface: 'QuantizationDebugInterface',
                 quantized_weight_modules_registry: Dict[Scope, torch.nn.Module],
                 quantized_inputs_modules_registry: Dict[Scope, torch.nn.Module],
                 weight_quantizers: Dict[QuantizationBuilder.WeightQuantizerKey, torch.nn.Module],
                 non_weight_quantizers: Dict[QuantizationBuilder.NonWeightQuantizerKey, torch.nn.Module]):
        super().__init__(target_model)
        self.debug_interface = debug_interface
        self.quantization_config = quantization_config

        self.quantized_weight_modules_registry = quantized_weight_modules_registry
        self.quantized_inputs_modules_registry = quantized_inputs_modules_registry
        self.weight_quantizers = weight_quantizers
        self.non_weight_quantizers = non_weight_quantizers
        self.all_quantizations = OrderedDict()
        self.all_quantizations.update(self.weight_quantizers)
        self.all_quantizations.update(self.non_weight_quantizers)
        self.is_distributed = False

    def distributed(self):
        self.is_distributed = True

    def initialize(self, data_loader=None, criterion=None):
        """
        For the quantization there are 2 types of initializations: range and precision.
        First method calculates per-layer activation statistics on training dataset in order to choose proper output
        range for quantization. Precision initialization happens based on measure - layers' sensitivity to
        perturbations. The measure is calculated by estimation of average trace of Hessian for modules using Hutchinson
        algorithm.
        Parameters for quantization algorithm:
            'data_loader' - provides an iterable over the given dataset, instance of 'torch.utils.data.DataLoader'
            'criterion' - loss function, instance of `torch.nn.modules.loss._Loss`,
        """

        initializer_config = self.quantization_config.get('initializer', {})
        init_range_config = initializer_config.get('range', {})
        num_init_steps = init_range_config.get('num_init_steps', 1)
        if num_init_steps < 0:
            raise AttributeError('Number of step to initialize must be >= 0')
        if num_init_steps > 0:
            global_init_type = init_range_config.get('type', 'mean_min_max')

            modules_to_init = OrderedDict()
            scope_overrides = self.quantization_config.get("scope_overrides", {})

            for class_type in QUANTIZATION_MODULES.registry_dict.values():
                quantization_type = class_type.__name__
                module_dict = get_all_modules_by_type(self._model, quantization_type)
                for scope, module in module_dict.items():
                    init_type = global_init_type
                    for overridden_scope in scope_overrides.keys():
                        if in_scope_list(str(scope), overridden_scope):
                            initializer_config = scope_overrides[overridden_scope].get('initializer', {})
                            init_type = initializer_config.get("type", global_init_type)
                    modules_to_init[str(scope)] = (module, init_type)

            # NOTE: Order of modules must be the same to correctly broadcast parameters (e.g. input_low
            # and input_range)
            modules_to_init = OrderedDict(sorted(modules_to_init.items()))

            runner = DataLoaderInitializeRunner(self._model, modules_to_init)
            if self.is_distributed:
                # Multi-process data loading heavily slows down collecting statistics. The best option, when data
                # fetching is done in the same process a DataLoader is initialized, i.e. num_workers should be 0.
                num_workers = data_loader.num_workers
                data_loader.num_workers = 0

                runner.run(data_loader, num_init_steps, self.is_distributed)
                data_loader.num_workers = num_workers
            else:
                runner.run(data_loader, num_init_steps, self.is_distributed)
            self._model.rebuild_graph()
        init_precision_config = initializer_config.get('precision', None)
        if init_precision_config:
            precision_init_type = init_precision_config.get('type', 'manual')

            params = self.quantization_config.get('activations', {})
            default_activation_bitwidth = params.get('bits', 8)
            params = self.quantization_config.get('weights', {})
            default_weight_bitwidth = params.get('bits', 8)

            init_impl = PrecisionInitializerFactory.create(precision_init_type)
            initializer = init_impl(self, init_precision_config, default_activation_bitwidth, default_weight_bitwidth,
                                    criterion, data_loader, self.is_distributed)
            initializer.apply_init()
            if is_main_process():
                nncf_logger.info('Bitwidth distribution\n{}'.format(self.get_bit_stats().draw()))

    def get_weights_activation_quantizers_pairs(self) -> List[Tuple[List[BaseQuantizer], BaseQuantizer]]:
        """
        finds all neighbour weight and input activation quantizers that share the same module (e.g. conv or linear).
        Single activation quantizer can be in pair with multiple neighbour weight quantizers, e.g. like in SqueezeNet,
        when two Convolutions share the same input activation.
        :return: list of pairs - (list of weight quantizers, activation quantizer)
        """
        pairs = []
        qimr = OrderedDict(sorted(self.quantized_inputs_modules_registry.items()))
        for _, quantized_module in qimr.items():
            weight_quantizer = None
            activation_quantizer = None
            for ops in quantized_module.pre_ops.values():
                if isinstance(ops, UpdateWeight):
                    weight_quantizer = ops.op
                if isinstance(ops, UpdateInputs):
                    activation_quantizer = ops.op
            pairs.append(([weight_quantizer], activation_quantizer))

        nncf_network = self._model
        nncf_graph = nncf_network.get_original_graph()
        non_weight_quantizers = {key: quantizer for key, quantizer in self.non_weight_quantizers.items() if
                                 not isinstance(key, QuantizationBuilder.InputQuantizerKey)}

        def traverse_graph(curr_nx_node_key: str, weight_quantizers: List[nn.Module]) -> Optional[List[nn.Module]]:
            nx_node = nncf_graph.get_nx_node_by_key(curr_nx_node_key)
            module_scope = nx_node[NNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR].scope_in_model
            module = nncf_network.get_module_by_scope(module_scope)
            if is_nncf_module(module):
                if hasattr(module, 'pre_ops'):
                    for ops in module.pre_ops.values():
                        if isinstance(ops, UpdateWeight):
                            weight_quantizers.append(ops.op)
            else:
                for succ_nx_node_key in nncf_graph.get_successors(curr_nx_node_key):
                    return traverse_graph(succ_nx_node_key, weight_quantizers)
            return weight_quantizers

        # pylint: disable=unnecessary-lambda
        for quantizer_key in sorted(non_weight_quantizers, key=lambda x: str(x)):
            activation_ctx = quantizer_key.ia_op_exec_context
            post_hooked_nx_node_key = nncf_graph.get_node_id_by_iap_context(activation_ctx)
            weight_quantizers = []
            for next_nx_node_key in nncf_graph.get_successors(post_hooked_nx_node_key):
                weight_quantizers = traverse_graph(next_nx_node_key, weight_quantizers)
            if weight_quantizers:
                activation_quantizer = self.non_weight_quantizers[quantizer_key]
                pairs.append((weight_quantizers, activation_quantizer))
        return pairs

    def get_bit_stats(self):
        table = Texttable()
        BITS = 'num_bits'
        WEIGHTS_RATIO = '% weights'
        ACTIVATIONS_RATIO = '% activations'
        TOTAL_RATIO = '% total'

        header = [BITS, WEIGHTS_RATIO, ACTIVATIONS_RATIO, TOTAL_RATIO]

        bits = set()
        num_all_quantizations = len(self.all_quantizations)
        for quantizer in self.all_quantizations.values():
            bits.add(quantizer.num_bits)

        bits_stat = {}
        for h in header:
            bits_stat[h] = {}
            for b in bits:
                bits_stat[h][b] = 0

        for quantizer in self.all_quantizations.values():  # type: BaseQuantizer
            num_bits = quantizer.num_bits
            bits_stat[TOTAL_RATIO][num_bits] += 1
            type_ = WEIGHTS_RATIO if quantizer.is_weights else ACTIVATIONS_RATIO
            bits_stat[type_][num_bits] += 1

        data = [header]

        for num_bits in bits:
            drow = {h: 0 for h in header}
            for column_name in header[1:]:
                drow[column_name] = (bits_stat[column_name][num_bits] / num_all_quantizations) * 100
            drow[BITS] = num_bits
            row = [drow[h] for h in header]
            data.append(row)
        table.add_rows(data)
        return table


class QuantizationDebugInterface(DebugInterface):
    QUANTIZERS_IN_NNCF_MODULES_TRACKER_NAME = 'quantized_modules'
    ACTIVATION_QUANTIZERS_TRACKER_NAME = 'activation_quantizers'
    FUNCTION_QUANTIZERS_TRACKER_NAME = 'function_quantizers'

    def __init__(self):
        self.call_trackers = {
            self.QUANTIZERS_IN_NNCF_MODULES_TRACKER_NAME: CallCountTracker(
                QuantizationDebugInterface.QUANTIZERS_IN_NNCF_MODULES_TRACKER_NAME),
            self.ACTIVATION_QUANTIZERS_TRACKER_NAME: CallCountTracker(
                QuantizationDebugInterface.ACTIVATION_QUANTIZERS_TRACKER_NAME),
            self.FUNCTION_QUANTIZERS_TRACKER_NAME: CallCountTracker(
                self.FUNCTION_QUANTIZERS_TRACKER_NAME)
        }
        self.graph_size = 0

        from nncf.debug import DEBUG_LOG_DIR
        self.dump_dir = Path(DEBUG_LOG_DIR) / Path("debug_dumps")
        self.dump_dir.mkdir(parents=True, exist_ok=True)
        self.scale_dump_dir = self.dump_dir / Path("scale")
        self.prop_graph_dump_dir = self.dump_dir / Path("quant_prop")
        if self.prop_graph_dump_dir.exists():
            shutil.rmtree(str(self.prop_graph_dump_dir))
        self.forward_call_count = 0
        self._strict_forward = False

    def init_actual(self, owner_model: NNCFNetwork):
        quantization_types = [class_type.__name__ for class_type in QUANTIZATION_MODULES.registry_dict.values()]
        quantizers_in_nncf_modules = owner_model.get_modules_in_nncf_modules_by_type(quantization_types)
        nncf_module_quantizations_id_list = [str(scope) for scope in
                                             quantizers_in_nncf_modules.keys()]  # type: List[str]

        activation_quantizer_id_list = owner_model.get_compression_modules_by_type(
            CompressionModuleType.ACTIVATION_QUANTIZER).keys()  # type: List[str]
        function_input_quantizer_id_list = owner_model.get_compression_modules_by_type(
            CompressionModuleType.FUNCTION_QUANTIZER).keys()  # type: List[str]
        self.call_trackers[self.QUANTIZERS_IN_NNCF_MODULES_TRACKER_NAME].init_with_key_list(
            nncf_module_quantizations_id_list)
        self.call_trackers[self.ACTIVATION_QUANTIZERS_TRACKER_NAME].init_with_key_list(
            activation_quantizer_id_list)
        self.call_trackers[self.FUNCTION_QUANTIZERS_TRACKER_NAME].init_with_key_list(
            function_input_quantizer_id_list)
        if self.scale_dump_dir.exists():
            shutil.rmtree(str(self.scale_dump_dir))
        self.scale_dump_dir.mkdir(parents=True, exist_ok=True)
        self._strict_forward = True

    def pre_forward_actions(self, module: 'NNCFNetwork'):
        self.reset_counters()

    def post_forward_actions(self, module: 'NNCFNetwork'):
        self.register_forward_call()
        # pylint:disable=protected-access
        ctx = module.get_tracing_context()
        self.set_graph_size(ctx.graph.get_nodes_count())

        quantization_types = [class_type.__name__ for class_type in QUANTIZATION_MODULES.registry_dict.values()]
        nncf_module_quantizations = module.get_modules_in_nncf_modules_by_type(
            quantization_types)  # type: Dict['Scope', nn.Module]

        for qm_scope, qm_module in nncf_module_quantizations.items():
            # Important - this will not work for DataParallel since it copies the
            # entire parent module for each thread and the `call_count` attributes
            # are incremented for thread local copies of `qm_module`, which are not
            # the same as the master copies of `qm_module` iterated over at this point
            self.register_quantizer_module_call(str(qm_scope), qm_module.call_count)
            self.dump_scale(qm_module.get_trainable_params(), str(qm_scope))
            qm_module.reset_call_counter()
        self.print_call_stats()

        call_dict = ctx.get_node_call_counter_dict()
        total_calls = sum(call_dict.values())
        nncf_logger.debug("{} nodes called out of total {}".format(total_calls,
                                                                   ctx.graph.get_nodes_count()))
        if self._strict_forward:
            for tracker in self.call_trackers.values():
                if tracker.get_never_called_keys():
                    # This will always trigger for DataParallel - disregard or disable debug mode
                    # for DataParallel runs
                    raise RuntimeError("{} has never called modules: {}!".format(
                        tracker.name, tracker.get_never_called_keys()))

    def dump_scale(self, quantizer_scale_params: Dict[str, torch.Tensor], quantizer_name: str):
        import re
        quantizer_normalized_name = re.sub(r'[^\w\-_\. ]', '_', quantizer_name)
        for scale_param_name, scale_param in quantizer_scale_params.items():
            fname = "{}_{}.txt".format(quantizer_normalized_name, scale_param_name)
            with open(str(self.scale_dump_dir / fname), "ba") as file:
                np.savetxt(file, scale_param.cpu().numpy())

    def reset_counters(self):
        for tracker in self.call_trackers.values():
            tracker.reset()

    def register_quantizer_module_call(self, key, counts=None):
        self.call_trackers[self.QUANTIZERS_IN_NNCF_MODULES_TRACKER_NAME].register_call(key, counts)

    def register_activation_quantize_call(self, key: str):
        self.call_trackers[self.ACTIVATION_QUANTIZERS_TRACKER_NAME].register_call(key)

    def register_function_quantizer_call(self, key: str):
        self.call_trackers[self.FUNCTION_QUANTIZERS_TRACKER_NAME].register_call(key)

    def print_call_stats(self):
        nncf_logger.debug(" Graph size: {} nodes".format(self.graph_size))
        for tracker in self.call_trackers.values():
            msg = " {} tracker:".format(tracker.name)
            msg += " {} total calls;".format(tracker.get_total_call_count())

            never_called = tracker.get_never_called_keys()
            if never_called:
                msg += " {} entries never called;".format(len(never_called))

            overcalled = tracker.get_overcalled_keys_with_call_counts()
            if overcalled:
                msg += " {} entries called more than once;".format(len(overcalled))
            nncf_logger.debug(msg)

    def set_graph_size(self, new_size):
        if new_size != self.graph_size:
            nncf_logger.debug('\n')
            nncf_logger.debug(
                " warning - graph size has changed from {} to {} since last forward".format(self.graph_size,
                                                                                            new_size))
        self.graph_size = new_size

    def register_forward_call(self):
        self.forward_call_count += 1

    def visualize_quantizer_propagation(self,
                                        prop_solver: QuantizerPropagationSolver,
                                        prop_graph: QuantizerPropagationStateGraph,
                                        iteration: str):
        self.prop_graph_dump_dir.mkdir(parents=True, exist_ok=True)
        fname = "quant_prop_iter_{}.dot".format(iteration)
        prop_solver.debug_visualize(prop_graph,
                                    self.prop_graph_dump_dir / Path(fname))

    def visualize_insertion_point_graph(self, insertion_point_graph: InsertionPointGraph):
        out_graph = nx.MultiDiGraph()
        for node_key, node in insertion_point_graph.nodes.items():
            if node[InsertionPointGraph.NODE_TYPE_NODE_ATTR] == InsertionPointGraphNodeType.INSERTION_POINT:
                insertion_point_data = node[InsertionPointGraph.INSERTION_POINT_DATA_NODE_ATTR]  # type: InsertionPoint
                label = "IP: {}".format(insertion_point_data.insertion_type)
                out_graph.add_node(node_key, label=label, color="red")
            elif node[InsertionPointGraph.NODE_TYPE_NODE_ATTR] == InsertionPointGraphNodeType.OPERATOR:
                out_graph.add_node(node_key)
            else:
                raise RuntimeError("Invalid InsertionPointGraph node!")
        for u, v in insertion_point_graph.edges:
            out_graph.add_edge(u, v)

        for node_key, node in insertion_point_graph.nodes.items():
            if node[InsertionPointGraph.NODE_TYPE_NODE_ATTR] == InsertionPointGraphNodeType.OPERATOR:
                for ip_node_key in node[InsertionPointGraph.ASSOCIATED_IP_NODE_KEYS_NODE_ATTR]:
                    out_graph.add_edge(node_key, ip_node_key, style="dashed", headport='e', tailport='e')

        nx.drawing.nx_pydot.write_dot(out_graph, self.dump_dir / Path("insertion_point_graph.dot"))
