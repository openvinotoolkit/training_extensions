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
import re
from collections import OrderedDict

import networkx as nx
import torch
from torch import nn

from .layers import QUANTIZATION_MODULES, BaseQuantizer
from ..dynamic_graph import InsertAfterNodeHook, context, get_context
from ..dynamic_graph.graph_matching import NodeExpression as N, search_all
from ..dynamic_graph.patch_pytorch import ignore_scope
from ..dynamic_graph.transform_graph import in_scope_list, replace_modules_by_nncf_modules
from ..dynamic_graph.utils import build_graph
from ..layers import NNCF_MODULES
from ..layer_utils import COMPRESSION_MODULES
from ..operations import UpdateWeight, UpdateInputs
from ..operator_names import VersionAgnosticNames
from ..utils import get_all_modules_by_type, get_state_dict_names_with_modules

logger = logging.getLogger(__name__)


@ignore_scope
class QuantizedNetwork(nn.Module):
    def __init__(self, module, quantize_module_creator_fn, inputs_shape=None, dummy_forward_fn=None,
                 ignored_scopes=None, target_scopes=None, quantize_inputs=True, quantize_outputs=False,
                 quantizable_subgraph_patterns=None):
        super().__init__()
        self.quantize_inputs = quantize_inputs
        self.quantize_outputs = quantize_outputs
        self.inputs_shape = inputs_shape
        self.ignored_scopes = ignored_scopes
        self.target_scopes = target_scopes
        self.module = module
        self.activation_quantizers = nn.ModuleDict()
        self.quantized_weight_modules = OrderedDict()
        self.quantized_activation_modules = OrderedDict()
        self.quantize_module_creator_fn = quantize_module_creator_fn
        self.quantizable_subgraph_patterns = quantizable_subgraph_patterns
        self.dummy_forward_fn = dummy_forward_fn

        device = next(module.parameters()).device

        self._scope = "orig"

        self.all_quantizations = OrderedDict()
        self._key_to_name = {}
        # all modules should be replaced prior to graph building
        self._replace_quantized_modules_by_nncf_modules(device)
        self._register_weight_quantization_operations(device)
        self._original_ctx = self.build_graph()
        self._scope = "quantized_graphs"
        self._ctx = get_context("quantized_graphs")
        self._register_activation_quantization_hooks(device)
        if self.quantize_inputs:
            self._register_input_quantization_operations(device)

        quantization_types = [class_type.__name__ for class_type in QUANTIZATION_MODULES.registry_dict.values()]
        self.all_quantizations = get_state_dict_names_with_modules(self, quantization_types)
        self.load_listener = LoadStateListener(self, self.all_quantizations)

    def forward(self, *args, **kwargs):
        with context(self._scope) as ctx:
            ctx.replica = self
            return self.module(*args, **kwargs)

    def _quantize_activation(self, ctx, activation, node):
        key = (node['type'], tuple(node['scope']), node['context'])
        name = self._key_to_name[key]
        logger.debug("Adding quantization after: {}".format(str(key)))
        replica = ctx.replica
        return replica.activation_quantizers[name](activation)

    def _replace_quantized_modules_by_nncf_modules(self, device):
        self.module = replace_modules_by_nncf_modules(self.module,
                                                      ignored_scopes=self.ignored_scopes,
                                                      target_scopes=self.target_scopes,
                                                      logger=logger)
        self.module = self.module.to(device)

    def _register_weight_quantization_operation(self, module_name, module, device):
        logger.info("Adding signed Weight quantizer in scope: {}".format(module_name))
        op = UpdateWeight(
            self.quantize_module_creator_fn(module_name, is_weights=True)
        ).to(device)
        module.register_pre_forward_operation(op)

    def _register_input_quantization_operation(self, module_name, module, device):
        quantizer = self.quantize_module_creator_fn(module_name, is_weights=False)

        logger.info("Adding {} Activation quantizer in scope: {}".format(
            "signed" if quantizer.signed else "unsigned", module_name
        ))

        module.register_pre_forward_operation(UpdateInputs(quantizer).to(device))

    def _register_weight_quantization_operations(self, device):
        modules = get_all_modules_by_type(self.module, NNCF_MODULES)

        for name, module in modules.items():
            if in_scope_list(name, self.ignored_scopes):
                logger.info("Ignored adding Weight quantizer in scope: {}".format(name))
                continue

            if self.target_scopes is None or in_scope_list(name, self.target_scopes):
                self.quantized_weight_modules[name] = module
                self._register_weight_quantization_operation(name, module, device)

    def _register_input_quantization_operations(self, device):
        # limitations:
        # graph is incorrectly split into subgraphs and there are no quantize layers before QuantizeMixin

        # todo move function to utils
        def _get_module_for_node(module, node_name):
            # module_name format is:
            # "node_id RootModType/ChildModType[child_name]/ChildModType[child_name]/..."

            # split node_id / node_uri
            node_name = node_name.split(' ', 1)[1]
            for part in node_name.split('/')[1:]:
                m = re.match(r'(\w+)(?:\[(.+)\])?', part)
                if not m:
                    logging.warning("could not parse node name: {}".format(node_name))
                    return None
                node_type, node_name = m.groups()
                if not node_name:
                    return module
                # pylint: disable=protected-access
                next_module = module._modules.get(node_name)
                if type(next_module).__name__ != node_type or not next_module:
                    return None
                module = next_module
            return module

        graph = self._original_ctx.graph
        graph_roots = [node for node, deg in graph.in_degree() if deg == 0]

        def get_first_noncompression_module(graph_node, graph):
            """ Gets the pre-op node immediately preceding the first non-COMPRESSION_MODULES node
                after `graph_node`.
                This is required in case there are multiple compression operations applied to the actual input node;
                for instance, in case of sparsity + quantization the input convolution might be preceded
                by 2 pre-ops - binary sparsity mask application and weight quantization
                """
            curr_m = _get_module_for_node(self.module, graph_node)
            if not isinstance(curr_m, tuple(COMPRESSION_MODULES.registry_dict.values())):
                return graph_node
            next_node = next(iter(graph.succ[graph_node]))
            m = _get_module_for_node(self.module, next_node)
            if isinstance(m, tuple(COMPRESSION_MODULES.registry_dict.values())):
                return get_first_noncompression_module(next_node, graph)
            return graph_node

        for idx, node in enumerate(graph_roots):
            graph_roots[idx] = get_first_noncompression_module(node, graph)

        inputs = []
        for node in graph_roots:
            # if the node is quantizer, we get its successor to get the input of original graph
            module = _get_module_for_node(self.module, node)
            if isinstance(module, tuple(QUANTIZATION_MODULES.registry_dict.values())):
                if graph.succ[node]:
                    next_node = next(iter(graph.succ[node]))
                    next_module = _get_module_for_node(self.module, next_node)
                    if next_module in self.quantized_weight_modules.values() and graph.in_degree()[next_node] == 1:
                        # Quantizer is the only input of the node
                        inputs.append(next_node)
            else:
                inputs.append(node)

        def _traverse_graph(node_name):
            module = _get_module_for_node(self.module, node_name)
            if module is None:
                return
            is_quantized_weight = module in self.quantized_weight_modules.values()
            if is_quantized_weight and module not in self.quantized_activation_modules.values():
                self.quantized_activation_modules[node_name] = module
                self._register_input_quantization_operation(node_name, module, device)

            if isinstance(module, tuple(QUANTIZATION_MODULES.registry_dict.values())) or is_quantized_weight:
                return

            for successor in graph.succ[node_name]:
                _traverse_graph(successor)

        for node in inputs:
            _traverse_graph(node)

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

    def _register_activation_quantization_hooks(self, device):
        pattern = self._make_custom_quantizable_subgraph_pattern()
        matches = search_all(pattern, self._original_ctx.graph)
        insertion_points = _find_insertion_points(self._original_ctx.graph, matches)
        original_graph_nodes = self._original_ctx.graph.nodes

        nodes_to_quantize = []
        for ip_name in insertion_points:
            ip_node = original_graph_nodes[ip_name]
            if not self.quantize_outputs and not list(self._original_ctx.graph.successors(ip_name)):
                logger.info("Ignored adding Activation Quantize "
                            "in scope (output scope, quantize_outputs=False): {}".format('/'.join(ip_node['scope'])))
                continue
            # ip_name = ip_name[ip_name.index(' ') + 1:]
            ip_name = ip_name[ip_name.index(' ') + 1:] + '_' + str(ip_node['context'])
            scope_name = '/'.join(ip_node['scope'])
            if in_scope_list(scope_name, self.ignored_scopes):
                logger.info("Ignored adding Activation quantizer in scope: {}".format('/'.join(ip_node['scope'])))
                continue

            if self.target_scopes is None or in_scope_list(scope_name, self.target_scopes):
                nodes_to_quantize.append(ip_node)
                ip_key = (ip_node['type'], tuple(ip_node['scope']), ip_node['context'])
                if ip_key in self._key_to_name:
                    raise RuntimeError(
                        "Ambiguous call to {fn} with call order {co} in current scope. "
                        "Cannot insert quantization hooks "
                        "automatically!".format(fn=ip_node['type'], co=ip_node['context'])
                    )
                self._key_to_name[ip_key] = ip_name

                assert ip_name not in self.activation_quantizers
                self.activation_quantizers[ip_name] = \
                    self.quantize_module_creator_fn(ip_node['scope'], is_weights=False).to(device)

                if isinstance(self.activation_quantizers[ip_name], BaseQuantizer):
                    logger.info("Adding {} Activation Quantize in scope: {}".format(
                        "signed" if self.activation_quantizers[ip_name].signed else
                        "unsigned", '/'.join(ip_node['scope'])
                    ))
                else:
                    logger.info("Adding Activation Binarize in scope: {}".format('/'.join(ip_node['scope'])))

        hook = InsertAfterNodeHook(self._quantize_activation, nodes_to_quantize)
        self._ctx.register_hook(hook, 'quantize_activations')

        # NOTE: Order of activations must be the same to correctly broadcast parameters (e.g. scales) in distributed
        # mode (see call of `_dist_broadcast_coalesced` in torch/nn/parallel/distributed.py for more details)
        # pylint: disable=protected-access
        self.activation_quantizers._modules = OrderedDict(sorted(self.activation_quantizers._modules.items()))

    def build_graph(self):
        ctx = build_graph(self, self._scope)
        # Reset initialization flags (`initialized`) for all quantization modules after dummy `load_state_dict` call.
        for module in self.all_quantizations.values():
            module.initialized = False
        return ctx

    def export(self, filename):
        self.eval()
        with torch.no_grad():
            param = next(self.parameters())
            input_shape = tuple([1] + list(self.inputs_shape)[1:])
            torch.onnx.export(self, param.new_zeros(input_shape), filename, verbose=True)


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

    pattern = linear_ops | eltwise | bn_relu | linear_ops + bn_relu | eltwise + bn_relu | single_ops
    return pattern


def _find_insertion_points(graph, matches):
    topological_order = {node: k for k, node in enumerate(nx.topological_sort(graph))}
    insertion_points = {max(match, key=topological_order.__getitem__) for match in matches}
    for match in matches:
        for node in match:
            if len(list(graph.successors(node))) > 1:
                insertion_points.add(node)

    return list(insertion_points)
