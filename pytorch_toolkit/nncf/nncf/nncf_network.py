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
from typing import List, Callable, Tuple, Dict, Optional

import functools
import networkx as nx
import torch
from copy import deepcopy
from torch import nn

from nncf.debug import CombinedDebugInterface, debuggable_forward, is_debug
from nncf.dynamic_graph.context import TracingContext
from nncf.dynamic_graph.graph import NNCFGraph, InputAgnosticOperationExecutionContext, OperationExecutionContext
from nncf.dynamic_graph.graph import ShapeIgnoringTensorMetaComparator
from nncf.dynamic_graph.graph_builder import GraphBuilder, PostGraphBuildActing, create_dummy_forward_fn, ModelInputInfo
from nncf.dynamic_graph.graph_matching import NodeExpression
from nncf.dynamic_graph.patch_pytorch import ignore_scope, nncf_model_input, MODEL_INPUT_OP_NAME
from nncf.dynamic_graph.operator_metatypes import OPERATOR_METATYPES
from nncf.dynamic_graph.transform_graph import replace_modules_by_nncf_modules
from nncf.hw_config import HWConfig
from nncf.layers import NNCF_MODULES
from nncf.quantization.layers import QUANTIZATION_MODULES
from nncf.utils import get_all_modules_by_type, get_state_dict_names_with_modules
from nncf.nncf_logger import logger as nncf_logger

MODEL_WRAPPED_BY_NNCF_ATTR_NAME = 'nncf_module'


class CompressionModuleType(Enum):
    FUNCTION_QUANTIZER = 0
    ACTIVATION_QUANTIZER = 1


@functools.total_ordering
class OperationPriority(Enum):
    DEFAULT_PRIORITY = 0
    SPARSIFICATION_PRIORITY = 2
    QUANTIZATION_PRIORITY = 11
    PRUNING_PRIORITY = 1

    def __lt__(self, other):
        # pylint: disable=comparison-with-callable
        return self.value < other.value


class InsertionType(Enum):
    OPERATOR_PRE_HOOK = 0
    OPERATOR_POST_HOOK = 1
    NNCF_MODULE_PRE_OP = 2
    NNCF_MODULE_POST_OP = 3

    def __eq__(self, other):
        # pylint: disable=comparison-with-callable
        if isinstance(other, InsertionType):
            return self.value == other.value
        return self.value == other


class InsertionInfo:
    def __init__(self, op_exec_context: OperationExecutionContext,
                 is_input=False,
                 is_output=False,
                 shape_to_operate_on=None):
        self.op_exec_context = op_exec_context  # type: OperationExecutionContext
        self.is_input = is_input
        self.is_output = is_output
        self.shape_to_operate_on = shape_to_operate_on

    def __eq__(self, other: 'InsertionInfo'):
        return self.op_exec_context == other.op_exec_context

    def __hash__(self):
        return self.op_exec_context.__hash__()


class InsertionPoint:
    def __init__(self, ia_op_exec_context: InputAgnosticOperationExecutionContext,
                 insertion_type: InsertionType):
        self.ia_op_exec_context = ia_op_exec_context
        self.insertion_type = insertion_type

    def __eq__(self, other: 'InsertionPoint'):
        return self.insertion_type == other.insertion_type and self.ia_op_exec_context == other.ia_op_exec_context

    def __str__(self):
        return str(self.insertion_type) + " " + str(self.ia_op_exec_context)

    def __hash__(self):
        return hash(str(self))


class InsertionCommand:
    def __init__(self, point: InsertionPoint, fn: Callable,
                 priority: OperationPriority = OperationPriority.DEFAULT_PRIORITY):
        self.insertion_point = point  # type: InsertionPoint
        self.fn = fn  # type: Callable
        self.priority = priority  # type: OperationPriority


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


class InsertionPointGraphNodeType(Enum):
    INSERTION_POINT = 0
    OPERATOR = 1


class InsertionPointGraph(nx.DiGraph):
    """
    This graph is built from the NNCFGraph representation of the model control flow graph and adds ephemeral
    "insertion point nodes" into the NNCF model graph representation corresponding to operator pre- and
    post-hooks. Module pre-op and post-op insertion points are currently not reflected here, but they are
    probably not required for quantizing activations, for which the quantizer propagation makes sense.
    This "insertion point graph" representation is useful for quantizer propagation and for referencing
    the compression algorithm hooks to the model operations to which they are applied to.
    """
    NODE_TYPE_NODE_ATTR = "node_type"
    INSERTION_POINT_DATA_NODE_ATTR = "insertion_point_data"
    IS_IN_NNCF_MODULE_NODE_ATTR = "is_in_nncf_module"
    REGULAR_NODE_REF_NODE_ATTR = "regular_node_ref"
    ASSOCIATED_IP_NODE_KEYS_NODE_ATTR = "associated_ip_node_keys"
    OPERATOR_METATYPE_NODE_ATTR = "op_meta"

    PRE_HOOK_ID_PREFIX = "PRE HOOK "  # NB: Do not use colon (':') in node keys! Causes trouble for .dot file export.
    POST_HOOK_ID_PREFIX = "POST HOOK "

    def __init__(self, model_nx_graph: nx.DiGraph):
        super().__init__()
        self._base_nx_graph = deepcopy(model_nx_graph)

        for node_key, node in self._base_nx_graph.nodes.items():
            attrs = {InsertionPointGraph.REGULAR_NODE_REF_NODE_ATTR: node,
                     InsertionPointGraph.NODE_TYPE_NODE_ATTR: InsertionPointGraphNodeType.OPERATOR,
                     InsertionPointGraph.ASSOCIATED_IP_NODE_KEYS_NODE_ATTR: set(),
                     InsertionPointGraph.OPERATOR_METATYPE_NODE_ATTR: None}
            self.add_node(node_key, **attrs)
        for from_node, to_node in self._base_nx_graph.edges:
            self.add_edge(from_node, to_node)

        # TODO: Add insertion points for module pre- and post-ops.
        # Should roughly look so: first, determine subsets of nodes belonging to each
        # separate NNCF module (via scope analysis), then for each subset find input/output
        # edges using a corresponding NNCFGraph function; add a pre-op insertion point node as the
        # sink for input edges and connect it to input edge destinations, then add a post-op
        # insertion point as the source of output edges and connect it to output edge origins.

        node_keys_working_set = [deepcopy(node_key) for node_key in self.nodes.keys()]
        for operator_node_key in node_keys_working_set:
            original_node = self.nodes[operator_node_key][InsertionPointGraph.REGULAR_NODE_REF_NODE_ATTR]
            ia_op_exec_context = original_node[NNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR].input_agnostic

            # Pre-hook insertion point nodes
            pre_hook_insertion_point = InsertionPoint(ia_op_exec_context,
                                                      InsertionType.OPERATOR_PRE_HOOK)
            attrs = {
                InsertionPointGraph.NODE_TYPE_NODE_ATTR: InsertionPointGraphNodeType.INSERTION_POINT,
                InsertionPointGraph.INSERTION_POINT_DATA_NODE_ATTR: pre_hook_insertion_point,
            }
            ip_node_key = self.get_pre_hook_node_key(str(operator_node_key))
            self.add_node(ip_node_key, **attrs)
            in_edges = list(self.in_edges(operator_node_key))
            for from_node_key, to_node_key in in_edges:
                self.remove_edge(from_node_key, to_node_key)
                self.add_edge(from_node_key, ip_node_key)
            self.add_edge(ip_node_key, operator_node_key)
            operator_node = self.nodes[operator_node_key]
            operator_node[InsertionPointGraph.ASSOCIATED_IP_NODE_KEYS_NODE_ATTR].add(ip_node_key)

            # Post-hook insertion point nodes
            post_hook_insertion_point = InsertionPoint(ia_op_exec_context,
                                                       InsertionType.OPERATOR_POST_HOOK)
            attrs = {
                InsertionPointGraph.NODE_TYPE_NODE_ATTR: InsertionPointGraphNodeType.INSERTION_POINT,
                InsertionPointGraph.INSERTION_POINT_DATA_NODE_ATTR: post_hook_insertion_point
            }
            ip_node_key = self.get_post_hook_node_key(str(operator_node_key))
            self.add_node(ip_node_key, **attrs)
            out_edges = list(self.out_edges(operator_node_key))
            for from_node_key, to_node_key in out_edges:
                self.remove_edge(from_node_key, to_node_key)
                self.add_edge(ip_node_key, to_node_key)
                # TODO: introduce separate insertion points for operator outputs if
                # the outputs are semantically different
            self.add_edge(operator_node_key, ip_node_key)
            operator_node = self.nodes[operator_node_key]
            operator_node[InsertionPointGraph.ASSOCIATED_IP_NODE_KEYS_NODE_ATTR].add(ip_node_key)

    def get_ip_graph_with_merged_hw_optimized_operations(self,
                                                         hw_config: Optional[HWConfig] = None) -> 'InsertionPointGraph':
        merged_ip_graph = deepcopy(self)
        pattern = self._get_mergeable_operator_patterns(hw_config)
        from nncf.dynamic_graph.graph_matching import search_all
        matches = search_all(self._base_nx_graph, pattern)
        for match in matches:
            if len(match) == 1:
                continue

            input_node_key = match[0]
            output_node_key = match[-1]
            in_edges = list(self.in_edges(input_node_key))
            out_edges = list(self.out_edges(output_node_key))

            assert len(in_edges) <= 1  # TODO: change to == 1 when input nodes are handled correctly

            if in_edges:
                in_edge_key = in_edges[0]
                in_edge_copy = deepcopy(self.edges[in_edge_key])
            out_edge_copies_dict = {}
            for out_edge_key in out_edges:
                out_edge_copies_dict[out_edge_key] = deepcopy(self.edges[out_edge_key])

            conserved_edges_list = out_edges
            if in_edges:
                conserved_edges_list.append(in_edge_key)

            merged_node_attrs = deepcopy(self.nodes[input_node_key])
            merged_node_attrs[InsertionPointGraph.ASSOCIATED_IP_NODE_KEYS_NODE_ATTR] = set()
            merged_node_key = ""
            for node_key in match:
                ip_node_keys = self.nodes[node_key][InsertionPointGraph.ASSOCIATED_IP_NODE_KEYS_NODE_ATTR]
                for ip_node_key in ip_node_keys:
                    should_keep_ip_node = False
                    for edge_key in conserved_edges_list:
                        if ip_node_key in edge_key:
                            should_keep_ip_node = True
                            break
                    if should_keep_ip_node:
                        merged_node_attrs[InsertionPointGraph.ASSOCIATED_IP_NODE_KEYS_NODE_ATTR].add(ip_node_key)
                    else:
                        merged_ip_graph.remove_node(ip_node_key)
                merged_ip_graph.remove_node(node_key)
                merged_node_key += node_key + '\n'

            merged_ip_graph.add_node(merged_node_key, **merged_node_attrs)
            if in_edges:
                merged_ip_graph.add_edge(in_edge_key[0], merged_node_key, **in_edge_copy)
            for out_edge_key, out_edge_attrs in out_edge_copies_dict.items():
                merged_ip_graph.add_edge(merged_node_key, out_edge_key[1], **out_edge_attrs)

        return merged_ip_graph

    @staticmethod
    def get_pre_hook_node_key(node_key: str):
        return InsertionPointGraph.PRE_HOOK_ID_PREFIX + node_key

    @staticmethod
    def get_post_hook_node_key(node_key: str):
        return InsertionPointGraph.POST_HOOK_ID_PREFIX + node_key

    def _get_mergeable_operator_patterns(self, hw_config: Optional[HWConfig] = None) -> NodeExpression:
        """Resulting pattern should have single input; the operation with inputs to
        quantize should be the input operation; outputs should only be produced by one output node."""
        # TODO: Implement "repeating expressions" so that any number of "mergeable" operations
        # immediately following a linear/convolutional/matrix op are merged into one block
        import nncf.dynamic_graph.patterns as p
        pattern = p.LINEAR_OPS + p.ANY_BN_RELU_COMBO | p.LINEAR_OPS + p.ELTWISE_UNIFORM_OPS
        return pattern

    def get_op_nodes_in_scope(self, scope: 'Scope') -> List:
        matching_ip_graph_op_nodes_list = []
        for node in self.nodes().values():
            if node[InsertionPointGraph.NODE_TYPE_NODE_ATTR] == InsertionPointGraphNodeType.OPERATOR:
                nncf_graph_node_ref = node[InsertionPointGraph.REGULAR_NODE_REF_NODE_ATTR]
                op_exec_context = nncf_graph_node_ref[NNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR]
                op_scope = op_exec_context.input_agnostic.scope_in_model
                if op_scope in scope:
                    matching_ip_graph_op_nodes_list.append(node)
        return matching_ip_graph_op_nodes_list

# pylint: disable=too-many-public-methods
@ignore_scope
class NNCFNetwork(nn.Module, PostGraphBuildActing):

    def __init__(self, module, input_infos: List[ModelInputInfo] = None,
                 dummy_forward_fn=None, scopes_without_shape_matching=None,
                 ignored_scopes=None, target_scopes=None):
        super().__init__()
        self.set_nncf_wrapped_model(module)
        self.input_infos = input_infos
        self.ignored_scopes = ignored_scopes
        self.target_scopes = target_scopes
        self._dummy_forward_fn = dummy_forward_fn
        self._nncf_module_scopes = []  # type: List[Scope]
        self.scopes_without_shape_matching = scopes_without_shape_matching
        self.debug_interface = CombinedDebugInterface() if is_debug() else None
        self._extra_module_types = []  # type: List[CompressionModuleType]
        # pylint:disable=line-too-long
        self._insertions_into_original_graph = {}  # type: Dict[InsertionPoint, List[Tuple[Callable, OperationPriority]]]

        device = next(module.parameters()).device

        # all modules should be replaced prior to graph building
        self._replace_modules_by_nncf_modules(device)

        _orig_context = TracingContext()
        _orig_graph_build_forward_fn = self._get_dummy_forward_fn_for_graph_building(with_input_tracing=True)

        self._graph_builder = GraphBuilder(_orig_graph_build_forward_fn)

        _orig_context.add_node_comparators([MODEL_INPUT_OP_NAME], ShapeIgnoringTensorMetaComparator())
        if self.scopes_without_shape_matching:
            _orig_context.add_node_comparators(scopes_without_shape_matching,
                                               ShapeIgnoringTensorMetaComparator())

        self._original_graph = self._graph_builder.build_graph(self.get_nncf_wrapped_model(), _orig_context)

        self._compressed_context = TracingContext()

        self._dummy_forward_fn = self._get_dummy_forward_fn_for_graph_building(with_input_tracing=False)

        self._compressed_context.add_node_comparators([MODEL_INPUT_OP_NAME], ShapeIgnoringTensorMetaComparator())
        if self.scopes_without_shape_matching:
            self._compressed_context.add_node_comparators(scopes_without_shape_matching,
                                                          ShapeIgnoringTensorMetaComparator())
        self._load_listener = None

        self._builders = []  # type: List['CompressionAlgorithmBuilder']

    @debuggable_forward
    def forward(self, *args, **kwargs):
        with self._compressed_context as ctx:  # type: TracingContext
            ctx.base_module_thread_local_replica = self
            arglist = list(args)
            for idx, tensor in enumerate(arglist):  # TODO: extend to all tensors in args/kwargs hierarchy
                if isinstance(tensor, torch.Tensor):
                    arglist[idx] = nncf_model_input(tensor)
            args = tuple(arglist)
            retval = self.get_nncf_wrapped_model()(*args, **kwargs)
        return retval

    def register_algorithm(self, builder: 'CompressionAlgorithmBuilder'):
        """Should be called during *builder*'s *apply_to* method, otherwise there will be no corresponding
        controller returned by the network on the *commit_compression_changes* stage"""
        self._builders.append(builder)

    # Cannnot use property syntax here, otherwise the wrapped module will end up
    # being twice in the same checkpoint with different prefixes
    def get_nncf_wrapped_model(self):
        return getattr(self, MODEL_WRAPPED_BY_NNCF_ATTR_NAME)

    def set_nncf_wrapped_model(self, value):
        setattr(self, MODEL_WRAPPED_BY_NNCF_ATTR_NAME, value)

    def get_modules_in_nncf_modules_by_type(self, types) -> Dict['Scope', nn.Module]:
        nncf_modules = self.get_nncf_modules()
        retval = {}
        for nncf_module_scope, nncf_module in nncf_modules.items():
            nncf_module_scope.pop()
            for relative_scope, target_module in get_all_modules_by_type(nncf_module, types).items():
                retval[nncf_module_scope + relative_scope] = target_module
        return retval

    def register_insertion_command(self, command: InsertionCommand):
        point = command.insertion_point
        if point not in self._insertions_into_original_graph:
            self._insertions_into_original_graph[point] = [(command.fn, command.priority)]
        else:
            self._insertions_into_original_graph[point].append((command.fn, command.priority))

    def commit_compression_changes(self) -> 'CompressionAlgorithmController':
        for insertion_point, fn_list_with_priority in self._insertions_into_original_graph.items():
            fn_list_with_priority = sorted(fn_list_with_priority, key=lambda x: x[1])
            self._insertions_into_original_graph[insertion_point] = fn_list_with_priority
            self._insert_at_point(insertion_point, [x[0] for x in fn_list_with_priority])

        if self.debug_interface is not None:
            self.debug_interface.init_actual(self)

        quantization_types = [class_type.__name__ for class_type in QUANTIZATION_MODULES.registry_dict.values()]
        all_quantizations = get_state_dict_names_with_modules(self, quantization_types)
        self._load_listener = LoadStateListener(self, all_quantizations)

        if not self._builders:
            from nncf.algo_selector import NoCompressionAlgorithmController
            return NoCompressionAlgorithmController(self)

        if len(self._builders) == 1:
            return self._builders[0].build_controller(self)

        from nncf.composite_compression import CompositeCompressionAlgorithmController
        composite_controller = CompositeCompressionAlgorithmController(self)
        for algo_builder in self._builders:
            composite_controller.add(algo_builder.build_controller(self))
        return composite_controller

    def _insert_at_point(self, point: InsertionPoint, fn_list: List[Callable]):
        if point.insertion_type == InsertionType.OPERATOR_PRE_HOOK:
            self._compressed_context.register_pre_hooks(fn_list, point.ia_op_exec_context)
        elif point.insertion_type == InsertionType.OPERATOR_POST_HOOK:
            self._compressed_context.register_post_hooks(fn_list, point.ia_op_exec_context)
        else:
            norm_target_scope = self._normalize_variable_recurrent_scope(point.ia_op_exec_context.scope_in_model)
            norm_nncf_scopes = [self._normalize_variable_recurrent_scope(x) for x in self._nncf_module_scopes]
            assert norm_target_scope in norm_nncf_scopes  # Required for proper Recurrent/VariableRecurrent addressing
            nncf_module = self.get_module_by_scope(point.ia_op_exec_context.scope_in_model)
            if point.insertion_type == InsertionType.NNCF_MODULE_PRE_OP:
                for fn in fn_list:
                    nncf_module.register_pre_forward_operation(fn)
            elif point.insertion_type == InsertionType.NNCF_MODULE_POST_OP:
                for fn in fn_list:
                    nncf_module.register_post_forward_operation(fn)

    def __getattr__(self, name):
        wrapped_module = super().__getattr__(MODEL_WRAPPED_BY_NNCF_ATTR_NAME)
        if hasattr(wrapped_module, name):
            return getattr(wrapped_module, name)
        return super().__getattr__(name)

    def get_graph(self) -> NNCFGraph:
        return self._compressed_context.graph

    def get_original_graph(self) -> NNCFGraph:
        return self._original_graph

    def get_tracing_context(self) -> TracingContext:
        return self._compressed_context

    def _get_dummy_forward_fn_for_graph_building(self, with_input_tracing):
        if self._dummy_forward_fn is None:
            return create_dummy_forward_fn(self.input_infos,
                                           with_input_tracing=with_input_tracing)
        return self._dummy_forward_fn

    def _replace_modules_by_nncf_modules(self, device):
        module, self._nncf_module_scopes = replace_modules_by_nncf_modules(self.get_nncf_wrapped_model(),
                                                                           ignored_scopes=self.ignored_scopes,
                                                                           target_scopes=self.target_scopes)
        self.set_nncf_wrapped_model(module.to(device))

    def get_nncf_module_scopes(self) -> List['Scope']:
        return self._nncf_module_scopes

    def get_nncf_modules(self) -> Dict['Scope', torch.nn.Module]:
        return get_all_modules_by_type(self.get_nncf_wrapped_model(), NNCF_MODULES)

    def rebuild_graph(self, *input_args):
        self._compressed_context.reset_graph()
        dummy_forward_fn = self._get_dummy_forward_fn_for_graph_building(with_input_tracing=False)
        builder = GraphBuilder(dummy_forward_fn)
        _ = builder.build_graph(self, self._compressed_context)

    def post_build_graph_actions(self):
        # Reset initialization flags (`initialized`) for all quantization modules
        # after dummy `load_state_dict` call.
        quantization_types = [class_type.__name__ for class_type in QUANTIZATION_MODULES.registry_dict.values()]
        all_quantizations = get_state_dict_names_with_modules(self, quantization_types)
        for module in all_quantizations.values():
            module.initialized = False

    def get_post_pattern_insertion_points(self, pattern: 'NNCFNodeExpression',
                                          omit_nodes_in_nncf_modules=False) -> List[InsertionInfo]:
        io_infos = self._original_graph.get_matching_nncf_graph_pattern_io_list(pattern)

        insertion_infos = []
        for io_info in io_infos:
            # The input/output is given in terms of edges, but the post-hooks are currently applied to
            # nodes. Multiple output edges in a pattern I/O info may originate from one and the same
            # node, and we have to ensure that these resolve into just one insertion point - thus the usage of "set".
            pattern_insertion_info_set = set()
            if len(io_info.output_edges) > 1:
                nncf_logger.debug("WARNING: pattern has more than one activation output")

            for nncf_node in io_info.output_nodes:
                pattern_insertion_info_set.add(InsertionInfo(nncf_node.op_exec_context,
                                                             is_output=True,
                                                             shape_to_operate_on=None))
                # TODO: determine output shapes for output nodes to enable per-channel quantization

            # Ignore input nodes in the pattern for now, rely on the _quantize_inputs functions.
            # TODO: handle input quantization here as well

            # Since this function is currently only used for activation quantization purposes via operator
            # post-hook mechanism, we may take any edge and it will point from the same node where we will have to
            # insert a quantizer later. However, in the future the output edges may refer to activation tensors
            # with different sizes, in which case we have to insert different per-channel quantizers to
            # accomodate different trainable params if there is a difference in the channel dimension.
            # Furthermore, currently there is no distinction for single tensor output to multiple nodes and
            # multiple tensor output to multiple nodes ("chunk" operation is an example of the latter).
            # The pattern may also have unexpected outputs from a node in the middle of the pattern (see
            # "densenet121.dot" for an example of this) - need to decide what to do with that in terms
            # of quantization.
            # TODO: address the issues above.

            for nncf_edge in io_info.output_edges:
                pattern_insertion_info_set.add(InsertionInfo(nncf_edge.from_node.op_exec_context,
                                                             is_output=False,
                                                             shape_to_operate_on=nncf_edge.tensor_shape))
            insertion_infos += list(pattern_insertion_info_set)

        insertion_infos = list(
            set(insertion_infos))  # Filter the overlapping insertion points from different matches (happens for GNMT)
        insertion_infos_filtered = []

        for info in insertion_infos:
            if omit_nodes_in_nncf_modules and self.is_scope_in_nncf_module_scope(info.op_exec_context.scope_in_model):
                continue
            insertion_infos_filtered.append(info)

        return insertion_infos_filtered

    def is_scope_in_nncf_module_scope(self, scope: 'Scope'):
        # TODO: optimize
        norm_nncf_scopes = [self._normalize_variable_recurrent_scope(x) for x in self._nncf_module_scopes]
        norm_op_scope = self._normalize_variable_recurrent_scope(scope)
        for nncf_scope in norm_nncf_scopes:
            if norm_op_scope in nncf_scope:
                return True
        return False

    def register_compression_module_type(self, compression_module_type: CompressionModuleType):
        attr_name = self._compression_module_type_to_attr_name(compression_module_type)
        if compression_module_type in self._extra_module_types:
            raise RuntimeError("Module type {} is already registered".format(compression_module_type))
        self.__setattr__(attr_name, nn.ModuleDict())
        self._extra_module_types.append(compression_module_type)

    def add_compression_module(self, module_key: str, module: nn.Module,
                               compression_module_type: CompressionModuleType):
        attr_name = self._compression_module_type_to_attr_name(compression_module_type)
        if compression_module_type not in self._extra_module_types:
            raise RuntimeError("Module type {} was not registered".format(compression_module_type))
        self.__getattr__(attr_name)[module_key] = module

    def get_compression_modules_by_type(self, compression_module_type: CompressionModuleType) -> nn.ModuleDict:
        attr_name = self._compression_module_type_to_attr_name(compression_module_type)
        if compression_module_type not in self._extra_module_types:
            raise RuntimeError("Module type {} was not registered".format(compression_module_type))
        return self.__getattr__(attr_name)

    @staticmethod
    def _compression_module_type_to_attr_name(compression_module_type: CompressionModuleType):
        """Required for backward compatibility with checkpoints that store function and activation
        quantizers directly under corresponding attributes of NNCFNetwork."""
        if compression_module_type == CompressionModuleType.FUNCTION_QUANTIZER:
            return "function_quantizers"
        if compression_module_type == CompressionModuleType.ACTIVATION_QUANTIZER:
            return "activation_quantizers"
        raise RuntimeError("Unknown extra module type")

    def sort_compression_modules(self, compression_module_type: CompressionModuleType):
        attr_name = self._compression_module_type_to_attr_name(compression_module_type)
        if compression_module_type not in self._extra_module_types:
            raise RuntimeError("Module type {} was not registered".format(compression_module_type))
        module_dict = self.__getattr__(attr_name)
        # pylint: disable=protected-access
        module_dict._modules = OrderedDict(sorted(module_dict._modules.items()))
        self.__setattr__(attr_name, module_dict)

    @staticmethod
    def _normalize_variable_recurrent_scope(scope: 'Scope'):
        """
        Two scopes pointing to an NNCF module that only differ in a Recurrent/VariableRecurrent/VariableRecurrentReverse
        scope element actually point to one and the same module.
        """
        ret_scope = scope.copy()
        for scope_element in ret_scope:
            if scope_element.calling_module_class_name in ["Recurrent", "VariableRecurrent",
                                                           "VariableRecurrentReverse"]:
                scope_element.calling_module_class_name = "NormalizedName_Recurrent"
        return ret_scope

    def do_dummy_forward(self, force_eval=False):
        """Attention: If run with force_eval=False, this may spoil the batchnorm statistics,
        and an eval run of the model will perform much worse than the train run. """
        if force_eval:
            train_mode = self.training
            self.eval()
        with torch.no_grad():
            self._dummy_forward_fn(self)
        if force_eval:
            if train_mode:
                self.train()

    def get_insertion_point_graph(self) -> InsertionPointGraph:
        ip_graph = InsertionPointGraph(self._original_graph.get_nx_graph_copy())

        # Mark IP graph operator nodes with associated op metatypes
        # Determining operator metatypes is more suited to occur at wrap_operator
        # stage, because it might be influenced by specific non-tensor function paramters,
        # but we have to inspect the containing module parameters as well, so the
        # TracingContext in wrap_operator would have to retain a reference to
        # the model that uses it. Since currently we do not need to inspect the
        # function arguments to determine the metatype, we can do this here, but
        # once we need to inspect the arguments, the code will have to be moved to
        # wrap_operator.

        for node_key in ip_graph.nodes:
            ip_graph_node = ip_graph.nodes[node_key]
            ip_graph_node_type = ip_graph_node[InsertionPointGraph.NODE_TYPE_NODE_ATTR]
            if ip_graph_node_type == InsertionPointGraphNodeType.OPERATOR:
                nncf_graph_node_ref = ip_graph_node[InsertionPointGraph.REGULAR_NODE_REF_NODE_ATTR]
                op_exec_context = nncf_graph_node_ref[NNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR]
                op_name = op_exec_context.operator_name
                scope = op_exec_context.scope_in_model
                op_arch = OPERATOR_METATYPES.get_operator_metatype_by_op_name(op_name)
                module = self.get_module_by_scope(scope)
                if module is not None:
                    subtype = op_arch.determine_subtype(containing_module=module)
                    if subtype is not None:
                        op_arch = subtype
                ip_graph_node[InsertionPointGraph.OPERATOR_METATYPE_NODE_ATTR] = op_arch
        return ip_graph

    def get_module_by_scope(self, scope: 'Scope') -> torch.nn.Module:
        curr_module = self.get_nncf_wrapped_model()
        for scope_element in scope[1:]:  # omit first scope element which corresponds to base module
            if scope_element.calling_field_name is None:
                # The module used is being created in-place every time and never stored in the model,
                # happens for nn.Softmax in BERT implementations.
                return None
            # pylint: disable=protected-access
            next_module = curr_module._modules.get(scope_element.calling_field_name)
            if next_module is None:
                raise RuntimeError("Could not find a {} module member in {} module of scope {} during node search"
                                   .format(scope_element.calling_field_name,
                                           scope_element.calling_module_class_name,
                                           str(scope)))
            curr_module = next_module
        return curr_module
