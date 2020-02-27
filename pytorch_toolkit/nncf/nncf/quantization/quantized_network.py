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

import functools
import logging
import operator
from collections import OrderedDict
from typing import List

from torch import nn

from nncf.algo_selector import create_dummy_forward_fn
from nncf.debug import is_debug, QuantizationDebugInterface, debuggable_forward
from nncf.dynamic_graph.context import OperatorInput
from nncf.dynamic_graph.graph_builder import GraphBuilder, PostGraphBuildActing
from nncf.dynamic_graph.utils import get_module_for_scope
from .layers import QUANTIZATION_MODULES, BaseQuantizer
from ..dynamic_graph import context, get_context
from ..dynamic_graph.graph import NNCFGraph, NNCFNode, InputAgnosticOperationExecutionContext, NNCFNodeExpression as N
from ..dynamic_graph.graph import ShapeIgnoringTensorMetaComparator
from ..dynamic_graph.patch_pytorch import ignore_scope, FUNCTIONS_TO_QUANTIZE, get_arg_positions_to_quantize
from ..dynamic_graph.transform_graph import in_scope_list, replace_modules_by_nncf_modules
from ..layer_utils import COMPRESSION_MODULES
from ..layers import NNCF_MODULES
from ..operations import UpdateWeight, UpdateInputs
from ..operator_names import VersionAgnosticNames
from ..utils import get_all_modules_by_type, get_state_dict_names_with_modules

logger = logging.getLogger(__name__)

MODULE_WRAPPED_BY_NNCF_ATTR_NAME = 'nncf_module'


@ignore_scope
class QuantizedNetwork(nn.Module, PostGraphBuildActing):
    def __init__(self, module, quantize_module_creator_fn, input_infos=None,
                 dummy_forward_fn=None, ignored_scopes=None, target_scopes=None, quantize_inputs=True,
                 quantize_outputs=False, quantizable_subgraph_patterns=None, scopes_without_shape_matching=None,
                 disable_function_quantization_hooks=False):
        super().__init__()
        self.set_nncf_wrapped_module(module)
        self.quantize_inputs = quantize_inputs
        self.quantize_outputs = quantize_outputs
        self.input_infos = input_infos
        self.ignored_scopes = ignored_scopes
        self.target_scopes = target_scopes
        self.activation_quantizers = nn.ModuleDict()
        self.function_quantizers = nn.ModuleDict()
        self.quantized_weight_modules = OrderedDict()
        self.quantized_activation_modules = OrderedDict()
        self.quantize_module_creator_fn = quantize_module_creator_fn
        self.quantizable_subgraph_patterns = quantizable_subgraph_patterns
        self._dummy_forward_fn = dummy_forward_fn
        self._nncf_module_scopes = []  # type: List[Scope]
        self.debug_interface = QuantizationDebugInterface() if is_debug() else None
        self.scopes_without_shape_matching = scopes_without_shape_matching

        device = next(module.parameters()).device

        self.all_quantizations = OrderedDict()
        self._processed_input_agnostic_op_exec_contexts = set()
        self._processed_function_quantizers = set()

        # all modules should be replaced prior to graph building
        self._replace_quantized_modules_by_nncf_modules(device)
        self._register_weight_quantization_operations(device)

        if self._dummy_forward_fn is None:
            self._dummy_forward_fn = create_dummy_forward_fn(self.input_infos)

        self._graph_builder = GraphBuilder(custom_forward_fn=self._dummy_forward_fn)

        self._context_name = "orig"
        if self.scopes_without_shape_matching:
            get_context(self._context_name).add_node_comparators(scopes_without_shape_matching,
                                                                 ShapeIgnoringTensorMetaComparator())

        self._original_graph = self._graph_builder.build_graph(self, self._context_name)

        self._context_name = "quantized_graphs"
        self._ctx = get_context("quantized_graphs")
        if self.scopes_without_shape_matching:
            get_context(self._context_name).add_node_comparators(scopes_without_shape_matching,
                                                                 ShapeIgnoringTensorMetaComparator())

        self._register_activation_quantization_hooks(device)
        if self.quantize_inputs:
            self._register_input_quantization_operations(device)

        if not disable_function_quantization_hooks:
            self._register_function_quantization_hooks(device)

        quantization_types = [class_type.__name__ for class_type in QUANTIZATION_MODULES.registry_dict.values()]
        self.all_quantizations = get_state_dict_names_with_modules(self, quantization_types)
        self.load_listener = LoadStateListener(self, self.all_quantizations)
        if self.debug_interface is not None:
            self.debug_interface.init_actual(self.all_quantizations.keys(), self.activation_quantizers.keys(),
                                             self.function_quantizers.keys())

    @debuggable_forward
    def forward(self, *args, **kwargs):
        with context(self._context_name) as ctx:  # type: TracingContext
            ctx.base_module_thread_local_replica = self
            retval = self.get_nncf_wrapped_module()(*args, **kwargs)
        return retval

    # Cannnot use property syntax here, otherwise the wrapped module will end up
    # being twice in the same checkpoint with different prefixes
    def get_nncf_wrapped_module(self):
        return getattr(self, MODULE_WRAPPED_BY_NNCF_ATTR_NAME)

    def set_nncf_wrapped_module(self, value):
        setattr(self, MODULE_WRAPPED_BY_NNCF_ATTR_NAME, value)

    def __getattr__(self, name):
        wrapped_module = super().__getattr__(MODULE_WRAPPED_BY_NNCF_ATTR_NAME)
        if hasattr(wrapped_module, name):
            return getattr(wrapped_module, name)
        return super().__getattr__(name)

    def get_quantized_graph(self) -> NNCFGraph:
        return self._ctx.graph

    def get_context_name(self) -> str:
        return self._context_name

    def _should_consider_scope(self, scope_str: str) -> bool:
        return (self.target_scopes is None or in_scope_list(scope_str, self.target_scopes)) \
               and not in_scope_list(scope_str, self.ignored_scopes)

    def _replace_quantized_modules_by_nncf_modules(self, device):
        module, self._nncf_module_scopes = replace_modules_by_nncf_modules(self.get_nncf_wrapped_module(),
                                                                           ignored_scopes=self.ignored_scopes,
                                                                           target_scopes=self.target_scopes,
                                                                           logger=logger)
        self.set_nncf_wrapped_module(module.to(device))

    def _register_weight_quantization_operation(self, module_name, module, device):
        logger.info("Adding signed Weight quantizer in scope: {}".format(module_name))
        op = UpdateWeight(
            self.quantize_module_creator_fn(module_name, is_weights=True)
        ).to(device)
        module.register_pre_forward_operation(op)

    def _register_input_quantization_operation(self, module_name, module, device):
        # Only use the shape of the 0-th input info specified in config. TODO: fix this
        input_shape = self.input_infos[0].shape if self.input_infos is not None else None
        quantizer = self.quantize_module_creator_fn(module_name, is_weights=False,
                                                    input_shape=input_shape)

        logger.info("Adding {} input quantizer in scope: {}".format(
            "signed" if quantizer.signed else "unsigned", module_name
        ))

        module.register_pre_forward_operation(UpdateInputs(quantizer).to(device))

    def _register_weight_quantization_operations(self, device):
        modules = get_all_modules_by_type(self.get_nncf_wrapped_module(), NNCF_MODULES)

        for name, module in modules.items():
            if not self._should_consider_scope(name):
                logger.info("Ignored adding Weight quantizer in scope: {}".format(name))
                continue

            self.quantized_weight_modules[name] = module
            self._register_weight_quantization_operation(name, module, device)

    def _register_input_quantization_operations(self, device):
        # limitations:
        # graph is incorrectly split into subgraphs and there are no quantize layers before QuantizeMixin

        graph_roots = self._original_graph.get_graph_roots()

        def get_first_noncompression_module_node_after(graph_node: NNCFNode, graph: NNCFGraph) -> NNCFNode:
            """ Gets the pre-op node immediately preceding the first non-COMPRESSION_MODULES node
                after `graph_node`.
                This is required in case there are multiple compression operations applied to the actual input node;
                for instance, in case of sparsity + quantization the input convolution might be preceded
                by 2 pre-ops - binary sparsity mask application and weight quantization
                """
            curr_m = get_module_for_scope(self.get_nncf_wrapped_module(), graph_node.op_exec_context.scope_in_model)
            if not isinstance(curr_m, tuple(COMPRESSION_MODULES.registry_dict.values())):
                return graph_node
            next_node_list = graph.get_next_nodes(graph_node)
            next_node = next(iter(next_node_list))  # not handling the branching case for now

            m = get_module_for_scope(self.get_nncf_wrapped_module(), next_node.op_exec_context.scope_in_model)
            if isinstance(m, tuple(COMPRESSION_MODULES.registry_dict.values())):
                return get_first_noncompression_module_node_after(next_node, graph)
            return graph_node

        for idx, node in enumerate(graph_roots):
            graph_roots[idx] = get_first_noncompression_module_node_after(node, self._original_graph)

        inputs = []
        for node in graph_roots:
            scope_str = str(node.op_exec_context.scope_in_model)
            # if the node is quantizer, we get its successor to get the input of original graph
            if self._should_consider_scope(scope_str):
                module = get_module_for_scope(self.get_nncf_wrapped_module(), node.op_exec_context.scope_in_model)
                if isinstance(module, tuple(QUANTIZATION_MODULES.registry_dict.values())):
                    next_node_list = self._original_graph.get_next_nodes(node)
                    if next_node_list:
                        next_node = next(iter(next_node_list))  # not handling the branching case for now
                        next_module = get_module_for_scope(self.get_nncf_wrapped_module(),
                                                           next_node.op_exec_context.scope_in_model)
                        if next_module in self.quantized_weight_modules.values() and \
                            self._original_graph.get_inputs_count(next_node) == 1:
                            # Quantizer is the only input of the node
                            inputs.append(next_node)
                else:
                    inputs.append(node)

        def _add_input_quantizers_traverser(node: NNCFNode) -> bool:
            module = get_module_for_scope(self.get_nncf_wrapped_module(), node.op_exec_context.scope_in_model)
            if module is None:
                return True
            is_quantized_weight = module in self.quantized_weight_modules.values()
            module_name = str(node.op_exec_context.scope_in_model)
            if is_quantized_weight and module not in self.quantized_activation_modules.values():
                self.quantized_activation_modules[module_name] = module
                self._register_input_quantization_operation(module_name, module, device)

            if isinstance(module, tuple(QUANTIZATION_MODULES.registry_dict.values())) or is_quantized_weight:
                return True
            return False

        for node in inputs:
            self._original_graph.traverse_graph(node, _add_input_quantizers_traverser)

    def _make_custom_quantizable_subgraph_pattern(self):
        full_pattern = _make_quantizable_subgraph_pattern()
        if self.quantizable_subgraph_patterns is not None:
            for pattern in self.quantizable_subgraph_patterns:
                if not isinstance(pattern, str):
                    custom_pattern = functools.reduce(operator.add,
                                                      [N(node) for node in pattern])
                else:
                    custom_pattern = N(pattern)
                full_pattern = full_pattern | custom_pattern
        return full_pattern

    class ActivationQuantizationHook:
        """Cannot simply register the quantizer module as a callable hook, since we need to call
        a thread-local version of the quantizer module during base module execution."""

        def __init__(self, context_name: str, ia_op_exec_context: InputAgnosticOperationExecutionContext,
                     debug_interface: QuantizationDebugInterface = None):
            self.context_name = context_name
            self.ia_op_exec_context = ia_op_exec_context
            self.debug_interface = debug_interface

        def __call__(self, *args, **kwargs):
            if self.debug_interface is not None:
                self.debug_interface.register_activation_quantize_call(str(self.ia_op_exec_context))
            replica = get_context(self.context_name).base_module_thread_local_replica
            return replica.activation_quantizers[str(self.ia_op_exec_context)](*args, **kwargs)

    def _register_activation_quantization_hooks(self, device):
        pattern = self._make_custom_quantizable_subgraph_pattern()
        insertion_point_nncf_nodes = self._original_graph.get_insertion_point_nodes_after_pattern(pattern)

        for ip_node in insertion_point_nncf_nodes:
            ia_op_exec_context = ip_node.op_exec_context.input_agnostic
            operator_scope_str = str(ia_op_exec_context)

            if not self.quantize_outputs and self._original_graph.is_output_node(ip_node):
                logger.info("Ignored adding Activation Quantize "
                            "in scope (output scope, quantize_outputs=False): {}".format(operator_scope_str))
                continue
            if not self._should_consider_scope(operator_scope_str):
                logger.info("Ignored adding Activation quantizer in scope: {}".format(operator_scope_str))
                continue

            if ia_op_exec_context in self._processed_input_agnostic_op_exec_contexts:
                raise RuntimeError(
                    "Ambiguous call to {fn} with call order {co} in current scope. "
                    "Cannot insert quantization hooks "
                    "automatically!".format(fn=ia_op_exec_context.operator_name, co=ia_op_exec_context.call_order)
                )
            self._processed_input_agnostic_op_exec_contexts.add(ia_op_exec_context)

            assert ia_op_exec_context not in self.activation_quantizers
            input_shape = ip_node.op_exec_context.tensor_metas[0].shape
            quantizer = self.quantize_module_creator_fn(operator_scope_str,
                                                        is_weights=False,
                                                        input_shape=input_shape).to(device)
            self.activation_quantizers[operator_scope_str] = quantizer

            if isinstance(quantizer, BaseQuantizer):
                logger.info("Adding {} Activation Quantize in scope: {}".format(
                    "signed" if quantizer.signed else
                    "unsigned", operator_scope_str
                ))
            else:
                logger.info("Adding Activation Binarize in scope: {}".format(operator_scope_str))

            self._ctx.register_post_hooks([self.ActivationQuantizationHook(self._context_name, ia_op_exec_context,
                                                                           self.debug_interface), ],
                                          ia_op_exec_context)

        # NOTE: Order of activations must be the same to correctly broadcast parameters (e.g. scales) in distributed
        # mode (see call of `_dist_broadcast_coalesced` in torch/nn/parallel/distributed.py for more details)
        # pylint: disable=protected-access
        self.activation_quantizers._modules = OrderedDict(sorted(self.activation_quantizers._modules.items()))

    def rebuild_graph(self, *input_args):
        ctx = get_context(self._context_name)
        ctx.reset_graph()
        _ = self._graph_builder.build_graph(self, self._context_name)

    def post_build_graph_actions(self):
        # Reset initialization flags (`initialized`) for all quantization modules
        # after dummy `load_state_dict` call.
        for module in self.all_quantizations.values():
            module.initialized = False

    class FunctionQuantizerKey:
        def __init__(self, ia_op_exec_context: InputAgnosticOperationExecutionContext, input_arg_idx: int):
            self.ia_op_exec_context = ia_op_exec_context
            self.input_arg_idx = input_arg_idx

        def __str__(self):
            return str(self.ia_op_exec_context) + "_input" + str(self.input_arg_idx)

        def __hash__(self):
            return hash((self.ia_op_exec_context, self.input_arg_idx))

    class FunctionQuantizationPreHook:
        """Cannot simply register the quantizer module as a callable hook, since we need to call
        a thread-local version of the quantizer module during base module execution."""

        def __init__(self, context_name: str, func_in_quant_info: 'FunctionQuantizationInfo',
                     debug_interface: QuantizationDebugInterface = None):
            self.context_name = context_name
            self.func_in_quant_info = func_in_quant_info
            self.debug_interface = debug_interface

        def __call__(self, op_inputs: OperatorInput):
            quantizer_dict_key = str(self.func_in_quant_info)
            if self.debug_interface is not None:
                self.debug_interface.register_function_quantizer_call(quantizer_dict_key)
            replica = get_context(self.context_name).base_module_thread_local_replica
            idx = self.func_in_quant_info.input_arg_idx
            op_inputs.op_args[idx] = replica.function_quantizers[quantizer_dict_key](op_inputs.op_args[idx])
            return op_inputs

    def _register_function_quantization_hooks(self, device):
        if not FUNCTIONS_TO_QUANTIZE:
            return
        pattern = N(FUNCTIONS_TO_QUANTIZE[0].name)
        for i in range(1, len(FUNCTIONS_TO_QUANTIZE)):
            pattern |= N(FUNCTIONS_TO_QUANTIZE[i].name)

        insertion_points = self._original_graph.get_insertion_point_nodes_after_pattern(pattern)

        non_shadowed_insertion_points = []
        for ip_node in insertion_points:
            is_function_in_nncf_module = False
            for nncf_scope in self._nncf_module_scopes:
                if ip_node.op_exec_context.scope_in_model in nncf_scope:
                    is_function_in_nncf_module = True
            if is_function_in_nncf_module:
                continue
            non_shadowed_insertion_points.append(ip_node)

        for ip_node in non_shadowed_insertion_points:
            ia_op_exec_context = ip_node.op_exec_context.input_agnostic
            scope_str = str(ia_op_exec_context.scope_in_model)

            if not self._should_consider_scope(scope_str):
                logger.info("Ignored adding function input quantizer in scope: {}".format(scope_str))
                continue

            function_arg_positions_to_quantize = get_arg_positions_to_quantize(ia_op_exec_context.operator_name)
            assert function_arg_positions_to_quantize is not None, "Function with inputs to be quantized has " \
                                                                   "no info struct registered in " \
                                                                   "QUANTIZED_INPUT_FUNCTIONS!"

            pre_hooks_to_register = []
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
                assert ip_arg_quant_name not in self.function_quantizers
                input_shape = ip_node.op_exec_context.tensor_metas[0].shape
                self.function_quantizers[ip_arg_quant_name] = \
                    self.quantize_module_creator_fn(scope_str, is_weights=False,
                                                    input_shape=input_shape).to(device)

                logger.info("Adding {} Function Quantize: {}".format(
                    "signed" if self.function_quantizers[ip_arg_quant_name].signed else
                    "unsigned", ip_arg_quant_name))
                pre_hooks_to_register.append(self.FunctionQuantizationPreHook(self._context_name,
                                                                              ip_arg_quant_key,
                                                                              self.debug_interface))
            self._ctx.register_pre_hooks(pre_hooks_to_register, ia_op_exec_context)

        # NOTE: Order of input quantizers must be the same to correctly broadcast parameters (e.g. scales) in
        # distributed mode (see call of `_dist_broadcast_coalesced` in torch/nn/parallel/distributed.py for more
        # details) pylint: disable=protected-access
        self.function_quantizers._modules = OrderedDict(sorted(self.function_quantizers._modules.items()))


class LoadStateListener:
    """
        Resets the initialization flags (`initialized`) for all quantization modules on `load_state_dict` call.
        These flags are used to update not loaded params (from checkpoint or model's state)
        on initialization stage of algorithm.
        Flags reset is required on each call of `load_state_dict`, because internal method (`build_graph`)
        restores model state by calling this method.
    """

    def __init__(self, model, all_quantizations):
        for prefix, module in all_quantizations.items():
            module.state_dict_name = prefix
        # pylint: disable=protected-access
        self.hook = model._register_load_state_dict_pre_hook(
            functools.partial(self.hook_fn, quantize_modules=all_quantizations.values()))

    def hook_fn(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs,
                quantize_modules):
        for module in quantize_modules:
            module.initialized = False

    def close(self):
        self.hook.remove()


def _make_quantizable_subgraph_pattern():
    linear_ops = N('linear') | N('conv2d') | N('conv_transpose2d') | N('conv3d') | N('conv_transpose3d')
    relu = N(VersionAgnosticNames.RELU) | N('hardtanh')
    bn = N('batch_norm') | N('batch_norm3d')
    bn_relu = bn + relu | relu + bn | bn | relu
    pooling = N('adaptive_avg_pool2d') | N('adaptive_avg_pool3d') | N('avg_pool2d') | N('avg_pool3d')
    activations = N('elu') | N('elu_') | N('prelu') | N('sigmoid')
    single_ops = activations | pooling | N('mean')
    eltwise = N('__iadd__') | N('__add__') | N('__mul__')
    matmul = N('matmul') | N('bmm')

    pattern = linear_ops | eltwise | bn_relu | linear_ops + bn_relu | eltwise + bn_relu | single_ops | matmul
    return pattern
