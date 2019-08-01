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

import logging
import re
from collections import OrderedDict

import networkx as nx
import torch
from torch import nn

from .layers import Quantize
from ..dynamic_graph import InsertAfterNodeHook, context, get_context
from ..dynamic_graph.graph_matching import NodeExpression as N, search_all
from ..dynamic_graph.patch_pytorch import ignore_scope
from ..dynamic_graph.transform_graph import in_scope_list, replace_modules_by_nncf_modules
from ..dynamic_graph.utils import build_graph
from ..layers import NNCF_MODULES
from ..operations import UpdateWeight, UpdateInputs
from ..operator_names import VersionAgnosticNames
from ..utils import get_all_modules_by_type

logger = logging.getLogger(__name__)


@ignore_scope
class QuantizedNetwork(nn.Module):
    def __init__(self, module, inputs_shape=None, input_args=None, input_kwargs=None, bits=8, activation_signed=False,
                 symmetric=True, ignored_scopes=None, signed_activation_scope=None, quantize_inputs=False):
        super().__init__()
        self.quantize_inputs = quantize_inputs
        self.signed_activation_scopes = signed_activation_scope
        self.input_kwargs = input_kwargs
        self.input_args = input_args
        self.symmetric = symmetric
        self.inputs_shape = inputs_shape
        self.bits = bits
        self.activation_signed = activation_signed
        self.ignored_scopes = ignored_scopes
        self.module = module
        self.activation_quantizers = nn.ModuleDict()
        self.quantized_weight_modules = OrderedDict()
        self.quantized_activation_modules = OrderedDict()

        device = next(module.parameters()).device

        self._scope = "orig"

        self._key_to_name = {}
        # all modules should be replaced prior to graph building
        self._replace_quantized_modules_by_nncf_modules(device)
        self._register_weight_quantization_operations(device)
        self._original_ctx = build_graph(self, *self._make_input(module), self._scope)
        self._scope = "quantized_graphs"
        self._ctx = get_context("quantized_graphs")
        self._register_activation_quantization_hooks(device)
        if self.quantize_inputs:
            self._register_input_quantization_operations(device)

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
        self.module = replace_modules_by_nncf_modules(self.module, ignored_scope=self.ignored_scopes, logger=logger)
        self.module = self.module.to(device)

    def _register_weight_quantization_operation(self, module_name, module, device):
        logger.info("Adding signed Weight Quantize in scope: {}".format(module_name))
        op = UpdateWeight(
            Quantize(num_bits=self.bits, signed=True, qbias=(not self.symmetric), is_weights=True)
        ).to(device)
        module.register_pre_forward_operation(op)

    def _register_input_quantization_operation(self, module_name, module, device):
        logger.info("Adding {} Activation Quantize in scope: {}".format(
            "signed" if self.activation_signed else "unsigned", module_name
        ))
        op = UpdateInputs(
            Quantize(num_bits=self.bits, signed=self.activation_signed, qbias=False, is_weights=False)
        ).to(device)
        module.register_pre_forward_operation(op)

    def _register_weight_quantization_operations(self, device):
        modules = get_all_modules_by_type(self.module, NNCF_MODULES)

        for name, module in modules.items():
            if in_scope_list(name, self.ignored_scopes):
                logger.info("Ignored adding Weight Quantize in scope: {}".format(name))
                continue

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

        inputs = []
        for node in graph_roots:
            # if the node is Quantize, we get its successor to get the input of original graph
            module = _get_module_for_node(self.module, node)
            if isinstance(module, Quantize):
                if graph.succ[node]:
                    next_node = next(iter(graph.succ[node]))
                    next_module = _get_module_for_node(self.module, next_node)
                    if next_module in self.quantized_weight_modules.values() and graph.in_degree()[next_node] == 1:
                        # Quantize is the only input of the node
                        inputs.append(next_node)
            else:
                inputs.append(node)

        def _traverse_graph(node_name):
            module = _get_module_for_node(self.module, node_name)
            if module is None:
                return
            if module in self.quantized_weight_modules.values() and \
                module not in self.quantized_activation_modules.values():
                self.quantized_activation_modules[node_name] = module
                self._register_input_quantization_operation(node_name, module, device)

            if isinstance(module, Quantize) or module in self.quantized_weight_modules.values():
                return

            for successor in graph.succ[node_name]:
                _traverse_graph(successor)

        for node in inputs:
            _traverse_graph(node)

    def _register_activation_quantization_hooks(self, device):
        pattern = _make_quantizable_subgraph_pattern()
        matches = search_all(pattern, self._original_ctx.graph)

        insertion_points = _find_insertion_points(self._original_ctx.graph, matches)
        original_graph_nodes = self._original_ctx.graph.nodes
        nodes_to_quantize = []
        for ip_name in insertion_points:
            ip_node = original_graph_nodes[ip_name]
            # ip_name = ip_name[ip_name.index(' ') + 1:]
            ip_name = ip_name[ip_name.index(' ') + 1:] + '_' + str(ip_node['context'])
            if in_scope_list(ip_node['scope'], self.ignored_scopes):
                logger.info("Ignored adding Activation Quantize in scope: {}".format('/'.join(ip_node['scope'])))
                continue
            nodes_to_quantize.append(ip_node)
            ip_key = (ip_node['type'], tuple(ip_node['scope']), ip_node['context'])
            if ip_key in self._key_to_name:
                raise RuntimeError(
                    "Ambiguous call to {fn} with call order {co} in current scope. Cannot insert quantization hooks "
                    "automatically!".format(fn=ip_node['type'], co=ip_node['context'])
                )
            self._key_to_name[ip_key] = ip_name

            signed = self.activation_signed or in_scope_list(ip_node['scope'], self.signed_activation_scopes)
            logger.info("Adding {} Activation Quantize in scope: {}".format(
                "signed" if signed else "unsigned", '/'.join(ip_node['scope'])
            ))

            assert ip_name not in self.activation_quantizers
            self.activation_quantizers[ip_name] = \
                Quantize(qbias=(not self.symmetric), num_bits=self.bits, signed=signed).to(device)

        hook = InsertAfterNodeHook(self._quantize_activation, nodes_to_quantize, self.ignored_scopes)
        self._ctx.register_hook(hook, 'quantize_activations')

        # NOTE: Order of activations must be the same to correctly broadcast parameters (e.g. scales) in distributed
        # mode (see call of `_dist_broadcast_coalesced` in torch/nn/parallel/distributed.py for more details)
        # pylint: disable=protected-access
        self.activation_quantizers._modules = OrderedDict(sorted(self.activation_quantizers._modules.items()))

    def _make_input(self, module=None):
        if self.input_args is not None or self.input_kwargs is not None:
            input_args = self.input_args if self.input_args is not None else tuple()
            input_kwargs = self.input_kwargs if self.input_kwargs is not None else dict()
            return input_args, input_kwargs
        return (next(module.parameters()).new_empty(self.inputs_shape),), {}

    def build_graph(self):
        return build_graph(self, *self._make_input(self), self._scope)

    def export(self, filename):
        self.eval()
        with torch.no_grad():
            param = next(self.parameters())
            input_shape = tuple([1] + list(self.inputs_shape)[1:])
            torch.onnx.export(self, param.new_zeros(input_shape), filename, verbose=True)


def _make_quantizable_subgraph_pattern():
    conv = N('conv2d') | N('conv_transpose2d')
    relu = N(VersionAgnosticNames.RELU) | N('hardtanh')
    bn = N('batch_norm')
    add = N('__iadd__') | N('__add__')

    bn_relu = bn + relu | relu + bn | bn | relu

    pattern = conv | add | bn_relu | conv + bn_relu | add + bn_relu
    return pattern


def _find_insertion_points(graph, matches):
    topological_order = {node: k for k, node in enumerate(nx.topological_sort(graph))}
    return [max(match, key=topological_order.__getitem__) for match in matches]
