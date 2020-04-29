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
import warnings
from collections import deque
from copy import deepcopy
from enum import Enum
from typing import Dict, Tuple

import networkx as nx

from nncf.dynamic_graph.graph import OperationExecutionContext
# pylint: disable=wildcard-import
# pylint: disable=unused-wildcard-import
from nncf.dynamic_graph.operator_metatypes import *
from nncf.dynamic_graph.operator_metatypes import OPERATOR_METATYPES
from nncf.nncf_network import InsertionInfo, InsertionType, InsertionPointGraph, InsertionPointGraphNodeType
from nncf.quantization.layers import QuantizerConfig, QuantizationMode


class QuantizationTrait(Enum):
    """General, hardware-agnostic specifications for the relation of operators to quantization.
    Hardware-specific quantization configuration is handled elsewhere."""
    NON_QUANTIZABLE = -1
    QUANTIZATION_AGNOSTIC = 0
    INPUTS_QUANTIZABLE = 1


DEFAULT_QUANT_TRAIT_TO_OP_DICT = {
    QuantizationTrait.INPUTS_QUANTIZABLE: [
        Conv2dMetatype,
        Conv3dMetatype,
        ConvTranspose2dMetatype,
        ConvTranspose3dMetatype,
        LinearMetatype,
        HardTanhMetatype,
        TanhMetatype,
        ELUMetatype,
        PRELUMetatype,
        LayerNormMetatype,
        GELUMetatype,
        SigmoidMetatype,
        AddMetatype,
        MulMetatype,
        DivMetatype,
        ExpMetatype,
        ErfMetatype,
        MatMulMetatype,
        MeanMetatype,
        RoundMetatype
    ],
    QuantizationTrait.NON_QUANTIZABLE: [
        EmbeddingMetatype,
        SoftmaxMetatype
    ]
}  # type: Dict[QuantizationTrait, List[OperatorMetatype]]


class PropagatingQuantizer:
    """Used in conjunction with QuantizerPropagationStateGraph to keep track of
       the allowed quantization configs corresponding to the model operation node
       whose inputs it quantizes, and also of the nodes/edges in the model control
       graph that this quantizer affects. It should be moved against the data flow of
       the model, tracking the affected nodes and edges of
       QuantizerPropagationStateGraph. No actual quantization modules are used here,
       only the associated configs (such as bitwidths, modes, signed/unsigned
       attributes etc.)"""
    def __init__(self, id_: int, quant_configs: List[QuantizerConfig], init_location_node_key: str):
        self._potential_quant_configs = quant_configs  # type: List[QuantizerConfig]
        self.affected_edges = set()
        self.affected_ip_nodes = set()
        self.propagation_path = []
        self.current_location_node_key = init_location_node_key
        self.last_accepting_location_node_key = None
        self.id = id_

    def __eq__(self, other):
        return self.id == other.id

    @property
    def potential_quant_configs(self) -> List[QuantizerConfig]:
        return self._potential_quant_configs


class TransitionStatus(Enum):
    SHOULD_TRANSITION = 0
    SHOULD_MERGE = 1
    SHOULD_NOT_TRANSITION = 2


class PropagationStrategy(Enum):
    CONSERVATIVE = 0  # While propagating up through a downward-branching node,
                      # do not propagate if the propagation results in narrowing the list of
                      # quantization variants available to quantizers on neighbouring branches
    AGGRESSIVE = 1


QuantizerPropagationStateGraphNodeType = InsertionPointGraphNodeType


class QuantizerPropagationStateGraph(nx.DiGraph):
    """This class is based upon InsertionPointGraph and represents
       a"chessboard" for PropagatingQuantizer items.  It tracks the current state of
       quantizer propagation by associating the operator and insertion point nodes and
       edges to propagating quantizers, if any. It can move a propagating quantizer
       via own edges and mark its progress through the graph, which is required for
       resolving situations when multiple quantizers attempt to proceed via one and
       the same graph node/edge. This class is mainly operated upon by the
       QuantizerPropagationSolver objects."""
    PROPAGATING_QUANTIZER_NODE_ATTR = "propagating_quantizer"
    AFFECTING_PROPAGATING_QUANTIZERS_ATTR = "affecting_propagating_quantizers"
    QUANTIZATION_TRAIT_NODE_ATTR = "quantization_trait"
    ALLOWED_INPUT_QUANTIZATION_TYPES_NODE_ATTR = "allowed_input_quantization_types"
    OPERATOR_METATYPE_NODE_ATTR = "op_meta"
    INSERTION_POINT_DATA_NODE_ATTR = "insertion_point"
    NODE_TYPE_NODE_ATTR = "node_type"

    def __init__(self, ip_graph: InsertionPointGraph):
        super().__init__()
        ip_graph = deepcopy(ip_graph)
        self._created_prop_quantizer_counter = 0

        for node_key, node in ip_graph.nodes.items():
            qpg_node = {
                self.NODE_TYPE_NODE_ATTR: node[InsertionPointGraph.NODE_TYPE_NODE_ATTR]}
            if node[InsertionPointGraph.NODE_TYPE_NODE_ATTR] == InsertionPointGraphNodeType.INSERTION_POINT:
                qpg_node[self.PROPAGATING_QUANTIZER_NODE_ATTR] = None
                qpg_node[self.AFFECTING_PROPAGATING_QUANTIZERS_ATTR] = []
                qpg_node[self.INSERTION_POINT_DATA_NODE_ATTR] = node[
                    InsertionPointGraph.INSERTION_POINT_DATA_NODE_ATTR]
            elif node[InsertionPointGraph.NODE_TYPE_NODE_ATTR] == InsertionPointGraphNodeType.OPERATOR:
                qpg_node[self.ALLOWED_INPUT_QUANTIZATION_TYPES_NODE_ATTR] = set()
                qpg_node[
                    self.QUANTIZATION_TRAIT_NODE_ATTR] = QuantizationTrait.NON_QUANTIZABLE
                qpg_node[self.AFFECTING_PROPAGATING_QUANTIZERS_ATTR] = []
                qpg_node[self.OPERATOR_METATYPE_NODE_ATTR] = node[InsertionPointGraph.OPERATOR_METATYPE_NODE_ATTR]
            self.add_node(node_key, **qpg_node)

        for from_node, to_node, edge_data in ip_graph.edges(data=True):
            edge_data[self.AFFECTING_PROPAGATING_QUANTIZERS_ATTR] = []
            self.add_edge(from_node, to_node, **edge_data)

    def merge_quantizer_into_path(self, prop_quantizer: PropagatingQuantizer, path: List):
        curr_node = self.nodes[prop_quantizer.current_location_node_key]
        curr_node[QuantizerPropagationStateGraph.PROPAGATING_QUANTIZER_NODE_ATTR] = None
        surviving_quantizers = []  # type: List[PropagatingQuantizer]
        for from_node_key, to_node_key in path:
            edge = self.edges[from_node_key, to_node_key]
            potential_quantizers = edge[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]
            if potential_quantizers:
                surviving_quantizers = potential_quantizers
                break
            from_node = self.nodes[from_node_key]
            potential_quantizer = from_node[QuantizerPropagationStateGraph.PROPAGATING_QUANTIZER_NODE_ATTR]
            if potential_quantizer is None:
                if from_node[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]:
                    potential_quantizer = \
                        from_node[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR][0]

            if potential_quantizer is not None:
                prop_quantizer.affected_edges.add((from_node_key, to_node_key))
                edge[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR].append(prop_quantizer)
                surviving_quantizers.append(potential_quantizer)
                break

        if surviving_quantizers:
            for pq in surviving_quantizers:
                pq.affected_edges.update(prop_quantizer.affected_edges)
                for from_node_key, to_node_key in prop_quantizer.affected_edges:
                    from_node = self.nodes[from_node_key]
                    from_node_type = from_node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]
                    if from_node_type == QuantizerPropagationStateGraphNodeType.INSERTION_POINT:
                        # pylint:disable=line-too-long
                        self.nodes[from_node_key][QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR].append(pq)
            for affected_edge_tuple in prop_quantizer.affected_edges:
                edge = self.edges[affected_edge_tuple]
                affecting_quantizers = edge[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]
                for pq in surviving_quantizers:
                    affecting_quantizers.append(pq)
            self.remove_propagating_quantizer(prop_quantizer)
        else:
            raise RuntimeError("Not found surviving_quantizers!"
                               " Nodes quantized with quantizer #{} will be lost".format(prop_quantizer.id))

    def backtrack_propagation_until_accepting_location(self, prop_quantizer: PropagatingQuantizer) -> Optional[
            PropagatingQuantizer]:
        if prop_quantizer.last_accepting_location_node_key is None:
            # The quantizer was stillborn.
            self.remove_propagating_quantizer(prop_quantizer)
            return None

        curr_node_key = prop_quantizer.current_location_node_key
        curr_node = self.nodes[curr_node_key]
        curr_node[QuantizerPropagationStateGraph.PROPAGATING_QUANTIZER_NODE_ATTR] = None
        while prop_quantizer.current_location_node_key != prop_quantizer.last_accepting_location_node_key:
            from_node_key, to_node_key = prop_quantizer.propagation_path.pop()

            edge = self.edges[from_node_key, to_node_key]
            edge[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR].remove(prop_quantizer)
            prop_quantizer.affected_edges.remove((from_node_key, to_node_key))
            from_node = self.nodes[from_node_key]
            from_node_type = from_node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]
            if from_node_type == QuantizerPropagationStateGraphNodeType.INSERTION_POINT:
                from_node[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR].remove(prop_quantizer)
                prop_quantizer.affected_ip_nodes.remove(from_node_key)

            to_node = self.nodes[to_node_key]
            to_node_type = to_node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]
            if to_node_type == QuantizerPropagationStateGraphNodeType.INSERTION_POINT:
                prop_quantizer.current_location_node_key = to_node_key

        target_ip_node_key = prop_quantizer.current_location_node_key
        target_node = self.nodes[target_ip_node_key]
        target_node[QuantizerPropagationStateGraph.PROPAGATING_QUANTIZER_NODE_ATTR] = prop_quantizer
        return prop_quantizer

    def add_propagating_quantizer(self, qconf_list: List[QuantizerConfig], ip_node_key: str) -> PropagatingQuantizer:
        prop_quantizer = PropagatingQuantizer(self._get_next_prop_quantizer_id(), qconf_list, ip_node_key)

        ip_node = self.nodes[ip_node_key]
        ip_node[QuantizerPropagationStateGraph.PROPAGATING_QUANTIZER_NODE_ATTR] = prop_quantizer
        ip_node[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR].append(prop_quantizer)

        ip_type = ip_node[QuantizerPropagationStateGraph.INSERTION_POINT_DATA_NODE_ATTR].insertion_type

        if ip_type != InsertionType.OPERATOR_PRE_HOOK:
            # The insertion point key should immediately precede a quantizable op,
            # otherwise it is hard to determine affected node here (although possible)
            raise RuntimeError("Can only add propagating quantizers into pre-hook spots!")

        affected_op_node_key = next(self.successors(ip_node_key))
        affected_op_node = self.nodes[affected_op_node_key]

        affected_op_node[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR].append(prop_quantizer)

        initial_edge_key = (ip_node_key, affected_op_node_key)
        initial_edge = self.edges[initial_edge_key]
        initial_edge[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR].append(prop_quantizer)
        prop_quantizer.affected_edges.add(initial_edge_key)
        prop_quantizer.affected_ip_nodes.add(ip_node_key)
        return prop_quantizer

    def clone_propagating_quantizer(self, prop_quantizer: PropagatingQuantizer) -> PropagatingQuantizer:
        cloned_prop_quant = deepcopy(prop_quantizer)
        cloned_prop_quant.id = self._get_next_prop_quantizer_id()
        for edge_tuple in cloned_prop_quant.affected_edges:
            edge = self.edges[edge_tuple]
            edge[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR].append(cloned_prop_quant)
        for node_key in cloned_prop_quant.affected_ip_nodes:
            node = self.nodes[node_key]
            node[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR].append(cloned_prop_quant)
        return cloned_prop_quant

    def remove_propagating_quantizer(self, prop_quantizer: PropagatingQuantizer):
        for edge_tuple in prop_quantizer.affected_edges:
            edge = self.edges[edge_tuple]
            affecting_quantizers = edge[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]
            affecting_quantizers.remove(prop_quantizer)
        for node_key in prop_quantizer.affected_ip_nodes:
            node = self.nodes[node_key]
            affecting_quantizers = node[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]
            affecting_quantizers.remove(prop_quantizer)

        node_key = prop_quantizer.current_location_node_key
        self.nodes[node_key][QuantizerPropagationStateGraph.PROPAGATING_QUANTIZER_NODE_ATTR] = None
        prop_quantizer.affected_ip_nodes.clear()
        prop_quantizer.affected_edges.clear()

    def propagate_quantizer_via_path(self, prop_quantizer: PropagatingQuantizer, path: List) -> PropagatingQuantizer:
        curr_node_key = prop_quantizer.current_location_node_key
        curr_node = self.nodes[curr_node_key]
        existing_quantizer = curr_node[QuantizerPropagationStateGraph.PROPAGATING_QUANTIZER_NODE_ATTR]
        if existing_quantizer is not None and existing_quantizer.id == prop_quantizer.id:
            curr_node[QuantizerPropagationStateGraph.PROPAGATING_QUANTIZER_NODE_ATTR] = None
        for edge_tuple in path:
            edge = self.edges[edge_tuple]
            edge[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR].append(prop_quantizer)
            prop_quantizer.affected_edges.add(edge_tuple)
            prop_quantizer.propagation_path.append(edge_tuple)
            from_node_key = edge_tuple[0]
            from_node = self.nodes[from_node_key]
            from_node_type = from_node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]
            if from_node_type == QuantizerPropagationStateGraphNodeType.INSERTION_POINT:
                from_node[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR].append(prop_quantizer)
                prop_quantizer.affected_ip_nodes.add(from_node_key)
                if self._is_position_accepting(from_node_key):
                    prop_quantizer.last_accepting_location_node_key = from_node_key

        target_ip_node_key = path[-1][0]
        prop_quantizer.current_location_node_key = target_ip_node_key
        target_node = self.nodes[target_ip_node_key]
        target_node[QuantizerPropagationStateGraph.PROPAGATING_QUANTIZER_NODE_ATTR] = prop_quantizer
        return prop_quantizer

    def get_quantizable_op_nodes_immediately_dominated_by_node(self, node_key) -> List[str]:
        ret_node_key_list = []

        def recursive_helper(curr_node_key: str, target_node_list: List[str]):
            successors = self.successors(curr_node_key)
            for successor_key in successors:
                successor = self.nodes[successor_key]
                successor_node_type = successor[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]
                if successor_node_type == QuantizerPropagationStateGraphNodeType.OPERATOR:
                    trait = successor[QuantizerPropagationStateGraph.QUANTIZATION_TRAIT_NODE_ATTR]
                    if not trait == QuantizationTrait.QUANTIZATION_AGNOSTIC:
                        target_node_list.append(successor_key)
                        return
                recursive_helper(successor_key, target_node_list)

        recursive_helper(node_key, ret_node_key_list)
        return ret_node_key_list

    def get_paths_to_immediately_dominating_insertion_points(self, insertion_point_node_key: str) -> List[List]:
        """Paths are lists of edges."""
        paths = []

        def recursive_helper(curr_edge, curr_path, all_paths):
            curr_path.append(curr_edge)
            curr_node_key = curr_edge[0]
            curr_node = self.nodes[curr_node_key]
            curr_node_type = curr_node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]
            if curr_node_type == QuantizerPropagationStateGraphNodeType.INSERTION_POINT:
                all_paths.append(curr_path)
                return

            for in_edge in self.in_edges(curr_node_key):
                path_copy = deepcopy(curr_path)
                recursive_helper(in_edge, path_copy, all_paths)

        for in_edge in self.in_edges(insertion_point_node_key):
            recursive_helper(in_edge, [], paths)
        return paths

    def get_visualized_graph(self):
        out_graph = nx.DiGraph()
        for node_key, node in self.nodes.items():
            node_type = node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]
            if node_type == QuantizerPropagationStateGraphNodeType.INSERTION_POINT:
                insertion_point_data = node[
                    QuantizerPropagationStateGraph.INSERTION_POINT_DATA_NODE_ATTR]  # type: InsertionPoint
                label = "IP: {}".format(insertion_point_data.insertion_type)
                out_graph.add_node(node_key, label=label, color="red")
                if node[QuantizerPropagationStateGraph.PROPAGATING_QUANTIZER_NODE_ATTR] is not None:
                    prop_quantizer = node[
                        QuantizerPropagationStateGraph.PROPAGATING_QUANTIZER_NODE_ATTR]  # type: PropagatingQuantizer
                    quant_node_key = "Quantizer #{}".format(prop_quantizer.id)
                    quant_configs_str_list = [str(conf) for conf in prop_quantizer.potential_quant_configs]
                    sub_label = '[' + ',\n'.join(quant_configs_str_list) + ']'
                    quant_node_label = quant_node_key + '\n' + "T: {}\n".format(sub_label)
                    out_graph.add_node(quant_node_key,
                                       color="blue", label=quant_node_label)
                    out_graph.add_edge(quant_node_key, node_key,
                                       style="dashed")
            elif node_type == QuantizerPropagationStateGraphNodeType.OPERATOR:
                out_graph.add_node(node_key)
            else:
                raise RuntimeError("Invalid QuantizerPropagationStateGraph node!")
        for u, v in self.edges:
            edge = self.edges[u, v]
            attrs = {}
            affecting_quantizers = edge[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]
            if affecting_quantizers:
                label = ", ".join([str(pq.id) for pq in affecting_quantizers])
                attrs = {"color": "blue", "label": label}
            out_graph.add_edge(u, v, **attrs)
        return out_graph

    def _get_next_prop_quantizer_id(self):
        self._created_prop_quantizer_counter += 1
        return self._created_prop_quantizer_counter

    def _is_position_accepting(self, ip_node_key: str):
        node = self.nodes[ip_node_key]
        insertion_type = node[QuantizerPropagationStateGraph.INSERTION_POINT_DATA_NODE_ATTR].insertion_type
        if insertion_type == InsertionType.OPERATOR_POST_HOOK:
            return True
        return False


class QuantizerPropagationSolver:
    """Analyzes a fresh QuantizerPropagationStateGraph object according to HW
       configuration supplied in the initializer and produces the list of insertion
       commands that correspond to the final state of the quantizer propagation graph
       when the model has the most contol flow graph edges quantized according to HW
       capabilities."""

    DEFAULT_QUANTIZATION_TYPES = [QuantizerConfig(
        bits=8,
        mode=QuantizationMode.SYMMETRIC,
        signedness_to_force=None,
        per_channel=False)]

    def __init__(self, hw_config=None,
                 debug_interface: 'QuantizationDebugInterface' = None,
                 propagation_strategy: PropagationStrategy = PropagationStrategy.AGGRESSIVE):
        self._hw_config = hw_config
        self._debug_interface = debug_interface
        self._propagation_strategy = propagation_strategy  # TODO: determine from config
        self._operator_quantization_trait_map = self.get_operator_quantization_traits_map()
        self._operator_allowed_qconfigs_map = self._get_operator_qconfigs_map()
        self._active_propagating_quantizers_queue = deque()
        self._finished_propagating_quantizers = []  # type: List[PropagatingQuantizer]

    def run_on_ip_graph(self, ip_graph: InsertionPointGraph) -> Dict[InsertionInfo, Optional[List[QuantizerConfig]]]:
        """ The main function to be used on an InsertionPointGraph to produce
            the list of insertion commands and configs corresponding to the final quantized
            graph state."""
        quant_prop_graph = QuantizerPropagationStateGraph(ip_graph)
        quant_prop_graph = self.set_allowed_quantization_types_for_operator_nodes(quant_prop_graph)
        quant_prop_graph = self.setup_initial_quantizers(quant_prop_graph)
        iteration_counter = 0
        while self._active_propagating_quantizers_queue:
            prop_quantizer = self._active_propagating_quantizers_queue.pop()
            if self._debug_interface is not None:
                self._debug_interface.visualize_quantizer_propagation(self, quant_prop_graph, str(iteration_counter))
            quant_prop_graph = self.propagation_step(prop_quantizer, quant_prop_graph)
            iteration_counter += 1

        if self._debug_interface is not None:
            self._debug_interface.visualize_quantizer_propagation(self, quant_prop_graph, "final")

        retval = {}
        for finished_prop_quantizer in self._finished_propagating_quantizers:
            final_node_key = finished_prop_quantizer.current_location_node_key
            final_node = quant_prop_graph.nodes[final_node_key]
            insertion_point = final_node[
                QuantizerPropagationStateGraph.INSERTION_POINT_DATA_NODE_ATTR]  # type: InsertionPoint
            insertion_info = InsertionInfo(OperationExecutionContext(
                operator_name=insertion_point.ia_op_exec_context.operator_name,
                scope_in_model=insertion_point.ia_op_exec_context.scope_in_model,
                call_order=insertion_point.ia_op_exec_context.call_order,
                tensor_metas=[None]
            ))  # TODO: fix this, rethink InsertionInfo here and elsewhere

            retval[insertion_info] = finished_prop_quantizer.potential_quant_configs
        return retval

    def propagation_step(self, curr_prop_quantizer: PropagatingQuantizer,
                         quant_prop_graph: QuantizerPropagationStateGraph) -> QuantizerPropagationStateGraph:
        """Returns an updated curr_prop_quantizer state if the quantizer is not
           yet in its final (accepting) position, and None if the quantizer is in its
           final location.  The location before and after the step should correspond to
           some insertion point."""
        # TODO: full-fledged discrete finite automata approach? Switch to traversing a graph
        # consisting of insertion points only, with reversed edges holding associated operator data?
        curr_node_key = curr_prop_quantizer.current_location_node_key
        curr_node = quant_prop_graph.nodes[curr_prop_quantizer.current_location_node_key]
        curr_node_type = curr_node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]
        assert curr_node_type == QuantizerPropagationStateGraphNodeType.INSERTION_POINT

        # Assumption: paths are at most 2 edges - either one edge to neighbouring insertion point
        # or one edge to operation and next edge to its own neighbouring insertion point.
        paths = quant_prop_graph.get_paths_to_immediately_dominating_insertion_points(curr_node_key)
        if not paths:
            prop_quantizer = quant_prop_graph.backtrack_propagation_until_accepting_location(curr_prop_quantizer)
            if prop_quantizer is not None:
                self._finished_propagating_quantizers.append(prop_quantizer)
            return quant_prop_graph

        surviving_prop_quantizers = []

        prop_quantizers_to_process = [curr_prop_quantizer]
        for _ in range(1, len(paths)):
            additional_prop_quantizer = quant_prop_graph.clone_propagating_quantizer(curr_prop_quantizer)
            prop_quantizers_to_process.append(additional_prop_quantizer)

        pqs_and_paths = zip(paths, prop_quantizers_to_process)
        for path, prop_quantizer in pqs_and_paths:
            status = self.check_transition_via_path(prop_quantizer, path, quant_prop_graph)
            if status == TransitionStatus.SHOULD_NOT_TRANSITION:
                prop_quantizer = quant_prop_graph.backtrack_propagation_until_accepting_location(prop_quantizer)
                if prop_quantizer is not None:
                    self._finished_propagating_quantizers.append(prop_quantizer)
            elif status == TransitionStatus.SHOULD_TRANSITION:
                prop_quantizer = quant_prop_graph.propagate_quantizer_via_path(prop_quantizer, path)
                surviving_prop_quantizers.append(prop_quantizer)
            elif status == TransitionStatus.SHOULD_MERGE:
                # The surviving quantizer will have its "affected edges" set extended
                # by the corresponding set of the merged quantizer. The assumption
                # here is that the surviving quantizer should never have to cross
                # such a "merge point" while backtracking to an accepting location.

                quant_prop_graph.merge_quantizer_into_path(prop_quantizer, path)

        for prop_quantizer in surviving_prop_quantizers:
            self._active_propagating_quantizers_queue.appendleft(prop_quantizer)
        return quant_prop_graph

    def get_allowed_quantizer_configs_for_operator(self, quant_det_id: OperatorMetatype) -> List[QuantizerConfig]:
        return self._operator_allowed_qconfigs_map[quant_det_id]

    def set_allowed_quantization_types_for_operator_nodes(self, quant_prop_graph: QuantizerPropagationStateGraph):
        for node_key, node in quant_prop_graph.nodes.items():
            node_type = node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]
            if node_type == QuantizerPropagationStateGraphNodeType.OPERATOR:
                quant_det_id = node[QuantizerPropagationStateGraph.OPERATOR_METATYPE_NODE_ATTR]
                if quant_det_id is None:
                    warnings.warn("Unknown metatype for operator node: {}".format(node_key))
                    trait = QuantizationTrait.QUANTIZATION_AGNOSTIC
                else:
                    trait = self._operator_quantization_trait_map[quant_det_id]
                node[QuantizerPropagationStateGraph.QUANTIZATION_TRAIT_NODE_ATTR] = trait
                if trait == QuantizationTrait.INPUTS_QUANTIZABLE:
                    node[QuantizerPropagationStateGraph.ALLOWED_INPUT_QUANTIZATION_TYPES_NODE_ATTR] = \
                        self.get_allowed_quantizer_configs_for_operator(quant_det_id)
        return quant_prop_graph

    def get_operator_quantization_traits_map(self) -> Dict[OperatorMetatype, QuantizationTrait]:
        # TODO: ensure that there are no name collisions between ops in different torch subpackages with the same name
        retval = {}
        if self._hw_config is None:
            for op_meta in OPERATOR_METATYPES.registry_dict.values():
                retval[op_meta] = QuantizationTrait.QUANTIZATION_AGNOSTIC  # Default value
            for trait, meta_list in DEFAULT_QUANT_TRAIT_TO_OP_DICT.items():
                for op_meta in meta_list:  # type: OperatorMetatype
                    retval[op_meta] = trait
        else:
            op_meta_vs_qconfs_map = self._hw_config.get_metatype_vs_quantizer_configs_map()
            for op_meta, qconf_list in op_meta_vs_qconfs_map.items():
                if qconf_list is None:
                    trait = QuantizationTrait.QUANTIZATION_AGNOSTIC
                elif qconf_list:
                    trait = QuantizationTrait.INPUTS_QUANTIZABLE
                else:
                    trait = QuantizationTrait.NON_QUANTIZABLE
                retval[op_meta] = trait
        return retval

    def _get_operator_qconfigs_map(self) -> Dict[OperatorMetatype, List[QuantizerConfig]]:
        # TODO: ensure that there are no name collisions between ops in different torch subpackages with the same name
        retval = {}
        if self._hw_config is None:
            for op_meta in OPERATOR_METATYPES.registry_dict.values():
                retval[op_meta] = QuantizationTrait.QUANTIZATION_AGNOSTIC  # Default value
            for trait, meta_list in DEFAULT_QUANT_TRAIT_TO_OP_DICT.items():
                if trait == QuantizationTrait.INPUTS_QUANTIZABLE:
                    for op_meta in meta_list:  # type: OperatorMetatype
                        retval[op_meta] = self.DEFAULT_QUANTIZATION_TYPES
                else:
                    for op_meta in meta_list:  # type: OperatorMetatype
                        retval[op_meta] = []
        else:
            retval = self._hw_config.get_metatype_vs_quantizer_configs_map()
        return retval


    def debug_visualize(self, quant_prop_graph: QuantizerPropagationStateGraph, dump_path: str):
        out_graph = quant_prop_graph.get_visualized_graph()
        active_ids_str = ", ".join([str(pq.id) for pq in self._active_propagating_quantizers_queue])
        finished_ids_str = ", ".join([str(pq.id) for pq in self._finished_propagating_quantizers])
        out_graph.graph['graph'] = {"label": "Propagating quantizers: {}\n" \
                                             "Finished quantizers: {}".format(active_ids_str, finished_ids_str),
                                    "labelloc": "t"}
        nx.drawing.nx_pydot.write_dot(out_graph, dump_path)

    def setup_initial_quantizers(self,
                                 quant_prop_graph: QuantizerPropagationStateGraph) -> QuantizerPropagationStateGraph:
        """Determines the initial subset of the nodes that must be quantized
           and corresponding allowed quantization configs (possibly multiple) for each
           quantizer."""
        for node_key, node in quant_prop_graph.nodes.items():
            node_type = node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]
            if node_type == QuantizerPropagationStateGraphNodeType.OPERATOR:
                preds = list(quant_prop_graph.predecessors(node_key))
                if not preds:
                    continue  # TODO: remove this once module insertion points are included in the IP graph
                # Should be immediately preceded by an insertion point.
                pred_ip_key = preds[0]
                pred_node = quant_prop_graph.nodes[pred_ip_key]
                pred_node_type = pred_node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]
                assert pred_node_type == QuantizerPropagationStateGraphNodeType.INSERTION_POINT, \
                    "Invalid insertion point graph supplied for quantizer propagation!"

                if node[QuantizerPropagationStateGraph.QUANTIZATION_TRAIT_NODE_ATTR] in [
                        QuantizationTrait.NON_QUANTIZABLE,
                        QuantizationTrait.QUANTIZATION_AGNOSTIC]:
                    continue

                quant_det_id = node[QuantizerPropagationStateGraph.OPERATOR_METATYPE_NODE_ATTR]
                qconf_list = self.get_allowed_quantizer_configs_for_operator(quant_det_id)
                prop_quantizer = quant_prop_graph.add_propagating_quantizer(qconf_list, pred_ip_key)
                self._active_propagating_quantizers_queue.appendleft(prop_quantizer)

        return quant_prop_graph

    def check_branching_transition(self, quant_prop_graph: QuantizerPropagationStateGraph,
                                   prop_quantizer: PropagatingQuantizer,
                                   branching_node_key: str) -> Optional[TransitionStatus]:
        """If a propagating quantizer advances through a node that branches
           downwards, the branches neighbouring to the one that the propagating quantizer
           had just propagated from will have the precision of the quantizer imposed upon
           them.  This is not always desirable - we might want to keep some branches in
           higher precision than the others. For this reason, this function checks whether
           the quantizer may safely advance through a branching node based on the possible
           configs of the quantizers it might affect by doing so."""
        dom_op_node_keys = quant_prop_graph.get_quantizable_op_nodes_immediately_dominated_by_node(
            branching_node_key)
        master_possible_qconfigs = prop_quantizer.potential_quant_configs
        slave_possible_qconfigs_dict = {}
        for op_node_key in dom_op_node_keys:
            op_node = quant_prop_graph.nodes[op_node_key]
            affecting_prop_quantizers = op_node[
                QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]
            if not affecting_prop_quantizers:
                # The branch op is forced to be FP32 - should not proceed through the branch node.
                return TransitionStatus.SHOULD_NOT_TRANSITION
            slave_possible_qconfigs = affecting_prop_quantizers[0].potential_quant_configs
            slave_possible_qconfigs_dict[op_node_key] = slave_possible_qconfigs
        master_merged_qconfigs, \
        slave_merged_qconfigs_dict = self.get_merged_qconfigs(master_possible_qconfigs,
                                                              slave_possible_qconfigs_dict)
        if not master_merged_qconfigs:
            # This quantizer's precision does not encompass the precisions of quantizers
            # propagating through downward branches.
            return TransitionStatus.SHOULD_NOT_TRANSITION

        if self._propagation_strategy == PropagationStrategy.CONSERVATIVE:
            for op_node_key, slave_merged_qconfigs_list in slave_merged_qconfigs_dict.items():
                if len(slave_possible_qconfigs_dict[op_node_key]) != len(slave_merged_qconfigs_list):
                    return TransitionStatus.SHOULD_NOT_TRANSITION

        return None

    def check_transition_via_path(self, prop_quantizer: PropagatingQuantizer, path: List,
                                  quant_prop_graph: QuantizerPropagationStateGraph) -> TransitionStatus:
        """Determines which action should be taken regarding the
           prop_quantizer's propagation via path, which may be one of many possible
           propagation paths."""
        for from_node_key, to_node_key in path:
            from_node = quant_prop_graph.nodes[from_node_key]

            if len(list(quant_prop_graph.successors(from_node_key))) > 1:
                # If a quantizer simply passes up through a downward-branching node, it may spoil the
                # precision for operations on neighbouring branches. Consider a 4-bit quantizer rising
                # through a branch node and an 8-bit quantizer arriving at the same node later. Therefore,
                # prior to allowing the quantizer to pass through a branching node we need to ensure that
                # the precision of the quantizer is a superset of precisions of the first non-quantization agnostic
                # operations on each branch.
                status = self.check_branching_transition(quant_prop_graph,
                                                         prop_quantizer,
                                                         from_node_key)
                if status is not None:
                    return status

            from_node_type = from_node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]
            if from_node_type == QuantizerPropagationStateGraphNodeType.OPERATOR:
                trait = from_node[QuantizerPropagationStateGraph.QUANTIZATION_TRAIT_NODE_ATTR]
                if trait in [QuantizationTrait.NON_QUANTIZABLE,
                             QuantizationTrait.INPUTS_QUANTIZABLE]:
                    return TransitionStatus.SHOULD_NOT_TRANSITION
            edge = quant_prop_graph.edges[from_node_key, to_node_key]
            potential_quantizers = edge[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]
            if potential_quantizers:
                # Assuming that multiple affecting quantizers should all have the same quantization config
                # by construction
                if prop_quantizer.potential_quant_configs == potential_quantizers[0].potential_quant_configs:
                    return TransitionStatus.SHOULD_MERGE
                return TransitionStatus.SHOULD_NOT_TRANSITION

            from_node_type = from_node[QuantizerPropagationStateGraph.NODE_TYPE_NODE_ATTR]
            if from_node_type == QuantizerPropagationStateGraphNodeType.INSERTION_POINT:
                potential_quantizers = from_node[QuantizerPropagationStateGraph.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]
                if potential_quantizers:
                    # Affecting quantizers should have the same configs by construction, so we only
                    # check the first
                    if prop_quantizer.potential_quant_configs == potential_quantizers[0].potential_quant_configs:
                        return TransitionStatus.SHOULD_MERGE
        return TransitionStatus.SHOULD_TRANSITION

    def get_merged_qconfigs(self, master_potential_qconfigs_list: List[QuantizerConfig],
                            slave_potential_qconfigs_dict: Dict[str, List[QuantizerConfig]]) -> Tuple[
                                List[QuantizerConfig], Dict[str, QuantizerConfig]]:
        """Returns potential qconfigs lists for 'master' and 'slave' quantizers
        that are compatible with each other. Compatibility is decided in terms of
        master quantizer having configs which all have higher precision than all the
        slave potential quantizer configs."""
        final_master_merged_qconfigs_list = deepcopy(master_potential_qconfigs_list)
        curr_slave_merged_qconfigs_dict = deepcopy(slave_potential_qconfigs_dict)
        # TODO: implement variant solutions, i.e. for each set of resultant merged
        # master potential qconfig lists we have, in general, different merged slave potential
        # config lists. Currently greedy approach is used.
        for m_qconfig in master_potential_qconfigs_list:
            should_persist_slave_merged_qconfigs_dict = True
            candidate_slave_merged_qconfigs_dict = deepcopy(curr_slave_merged_qconfigs_dict)
            for node_key, s_qconfig_list in curr_slave_merged_qconfigs_dict.items():
                for s_qconfig in s_qconfig_list:
                    if m_qconfig < s_qconfig and s_qconfig in candidate_slave_merged_qconfigs_dict[node_key]:
                        candidate_slave_merged_qconfigs_dict[node_key].remove(s_qconfig)
            for _, s_qconfig_list in candidate_slave_merged_qconfigs_dict.items():
                if not s_qconfig_list:
                    # No options left for slave configs on one of the branches to accomodate the master
                    # config - this master config cannot be used to be merged into.
                    final_master_merged_qconfigs_list.remove(m_qconfig)
                    should_persist_slave_merged_qconfigs_dict = False
                    break
            if should_persist_slave_merged_qconfigs_dict:
                curr_slave_merged_qconfigs_dict = candidate_slave_merged_qconfigs_dict
        if not final_master_merged_qconfigs_list:
            return [], {}
        return final_master_merged_qconfigs_list, curr_slave_merged_qconfigs_dict

    def get_finished_propagating_quantizers(self):
        return self._finished_propagating_quantizers

    def get_active_propagating_quantizers_queue(self):
        return self._active_propagating_quantizers_queue
