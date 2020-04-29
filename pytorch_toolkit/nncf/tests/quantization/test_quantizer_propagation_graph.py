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
from copy import deepcopy
from typing import List, Tuple
from collections import namedtuple, Counter

import networkx as nx
import pytest

from nncf.nncf_network import InsertionPointGraph
from nncf.quantization.layers import QuantizerConfig
from nncf.quantization.quantizer_propagation import QuantizerPropagationStateGraph as QPSG, \
    QuantizerPropagationStateGraphNodeType, QuantizationTrait
from tests.test_nncf_network import get_two_branch_mock_model_graph, get_mock_nncf_node_attrs


def get_edge_paths(graph, start_node_key, finish_node_key) -> List[List[Tuple]]:
    node_paths = list(nx.all_simple_paths(graph, start_node_key, finish_node_key))
    edge_paths = []
    for path in node_paths:
        edge_paths.append([(path[i], path[i + 1]) for i in range(0, len(path) - 1)])
    return edge_paths


def get_edge_paths_for_propagation(graph, start_node_key, finish_node_key) -> List[List[Tuple]]:
    paths = get_edge_paths(graph, start_node_key, finish_node_key)
    return [list(reversed(path)) for path in paths]


class TestQuantizerPropagationStateGraph:
    @staticmethod
    @pytest.fixture()
    def mock_qp_graph():
        ip_graph = InsertionPointGraph(get_two_branch_mock_model_graph())
        yield QPSG(ip_graph)

    def test_build_quantizer_propagation_state_graph_from_ip_graph(self):
        ip_graph = InsertionPointGraph(get_two_branch_mock_model_graph())
        quant_prop_graph = QPSG(ip_graph)
        assert len(ip_graph.nodes) == len(quant_prop_graph.nodes)
        assert len(ip_graph.edges) == len(quant_prop_graph.edges)

        for ip_graph_node_key, ip_graph_node in ip_graph.nodes.items():
            qpg_node = quant_prop_graph.nodes[ip_graph_node_key]
            assert qpg_node[QPSG.NODE_TYPE_NODE_ATTR] == ip_graph_node[
                InsertionPointGraph.NODE_TYPE_NODE_ATTR]
            qpg_node_type = qpg_node[QPSG.NODE_TYPE_NODE_ATTR]
            if qpg_node_type == QuantizerPropagationStateGraphNodeType.INSERTION_POINT:
                assert qpg_node[QPSG.PROPAGATING_QUANTIZER_NODE_ATTR] is None
                assert not qpg_node[QPSG.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]
                assert qpg_node[QPSG.INSERTION_POINT_DATA_NODE_ATTR] == ip_graph_node[
                    InsertionPointGraph.INSERTION_POINT_DATA_NODE_ATTR]
            elif qpg_node_type == QuantizerPropagationStateGraphNodeType.OPERATOR:
                assert not qpg_node[QPSG.ALLOWED_INPUT_QUANTIZATION_TYPES_NODE_ATTR]
                assert qpg_node[QPSG.QUANTIZATION_TRAIT_NODE_ATTR] == QuantizationTrait.NON_QUANTIZABLE
                assert not qpg_node[QPSG.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]

        for from_node, to_node, edge_data in ip_graph.edges(data=True):
            qpg_edge_data = quant_prop_graph.edges[from_node, to_node]
            assert not qpg_edge_data[QPSG.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]
            for key, value in edge_data.items():
                assert qpg_edge_data[key] == value

    def test_add_propagating_quantizer(self, mock_qp_graph):
        ref_qconf_list = [QuantizerConfig(), QuantizerConfig(bits=6)]

        target_node_key = "F"
        target_ip_node_key = InsertionPointGraph.get_pre_hook_node_key(target_node_key)
        prop_quant = mock_qp_graph.add_propagating_quantizer(ref_qconf_list, target_ip_node_key)
        assert prop_quant.potential_quant_configs == ref_qconf_list
        assert prop_quant.current_location_node_key == target_ip_node_key
        assert prop_quant.affected_ip_nodes == {target_ip_node_key}
        assert prop_quant.last_accepting_location_node_key is None
        assert prop_quant.affected_edges == {(target_ip_node_key, target_node_key)}
        assert not prop_quant.propagation_path

        for node_key, node in mock_qp_graph.nodes.items():
            if node_key == target_node_key:
                assert node[QPSG.AFFECTING_PROPAGATING_QUANTIZERS_ATTR] == [prop_quant]
            elif node_key == target_ip_node_key:
                assert node[QPSG.PROPAGATING_QUANTIZER_NODE_ATTR] == prop_quant
            else:
                assert not node[QPSG.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]
                if node[QPSG.NODE_TYPE_NODE_ATTR] == QuantizerPropagationStateGraphNodeType.INSERTION_POINT:
                    assert node[QPSG.PROPAGATING_QUANTIZER_NODE_ATTR] is None

            target_ip_node = mock_qp_graph.nodes[target_ip_node_key]
            assert target_ip_node[QPSG.PROPAGATING_QUANTIZER_NODE_ATTR] == prop_quant

        for from_node, to_node, edge_data in mock_qp_graph.edges.data():
            if (from_node, to_node) == (target_ip_node_key, target_node_key):
                assert edge_data[QPSG.AFFECTING_PROPAGATING_QUANTIZERS_ATTR] == [prop_quant]
            else:
                assert not edge_data[QPSG.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]

        with pytest.raises(RuntimeError):
            _ = mock_qp_graph.add_propagating_quantizer(ref_qconf_list,
                                                        InsertionPointGraph.get_post_hook_node_key(target_node_key))

    START_IP_NODES_AND_PATHS_TO_DOMINATING_IP_NODES = [
        # Non-branching case - starting from "E" pre-hook
        (InsertionPointGraph.get_pre_hook_node_key("E"),
         [[(InsertionPointGraph.get_post_hook_node_key("C"), InsertionPointGraph.get_pre_hook_node_key("E"))]]),

        # Non-branching case - starting from "C" post-hook
        (InsertionPointGraph.get_post_hook_node_key("C"),
         [[("C", InsertionPointGraph.get_post_hook_node_key("C")),
           (InsertionPointGraph.get_pre_hook_node_key("C"), "C")]]),

        # Branching case - starting from "F" pre-hook
        (InsertionPointGraph.get_pre_hook_node_key("F"),
         [[(InsertionPointGraph.get_post_hook_node_key("D"), InsertionPointGraph.get_pre_hook_node_key("F"))],
          [(InsertionPointGraph.get_post_hook_node_key("E"), InsertionPointGraph.get_pre_hook_node_key("F"))]]),

    ]

    @staticmethod
    @pytest.fixture(params=START_IP_NODES_AND_PATHS_TO_DOMINATING_IP_NODES)
    def start_ip_node_and_path_to_dominating_node(request):
        return request.param

    def test_get_paths_to_immediately_dominating_insertion_points(self, start_ip_node_and_path_to_dominating_node,
                                                                  mock_qp_graph):
        start_node = start_ip_node_and_path_to_dominating_node[0]
        ref_paths = start_ip_node_and_path_to_dominating_node[1]
        test_paths = mock_qp_graph.get_paths_to_immediately_dominating_insertion_points(start_node)
        def get_cat_path_list(path_list: List[List[Tuple[str, str]]]):
            str_paths = [[str(edge[0]) +' -> ' + str(edge[1]) for edge in path] for path in path_list]
            cat_paths = [';'.join(path) for path in str_paths]
            return cat_paths

        assert Counter(get_cat_path_list(ref_paths)) == Counter(get_cat_path_list(test_paths))


    START_TARGET_NODES = [
        (InsertionPointGraph.get_pre_hook_node_key("H"),
         InsertionPointGraph.get_post_hook_node_key("G")),

        (InsertionPointGraph.get_pre_hook_node_key("H"),
         InsertionPointGraph.get_pre_hook_node_key("F")),

        (InsertionPointGraph.get_pre_hook_node_key("F"),
         InsertionPointGraph.get_pre_hook_node_key("E")),

        (InsertionPointGraph.get_pre_hook_node_key("F"),
         InsertionPointGraph.get_post_hook_node_key("B")),
    ]

    @staticmethod
    @pytest.fixture(params=START_TARGET_NODES)
    def start_target_nodes(request):
        return request.param

    @pytest.mark.dependency(name="propagate_via_path")
    def test_quantizers_can_propagate_via_path(self, start_target_nodes, mock_qp_graph):
        start_ip_node_key = start_target_nodes[0]
        target_ip_node_key = start_target_nodes[1]

        # From "target" to "start" since propagation direction is inverse to edge direction
        rev_paths = get_edge_paths_for_propagation(mock_qp_graph, target_ip_node_key, start_ip_node_key)

        for path in rev_paths:
            working_graph = deepcopy(mock_qp_graph)
            ref_prop_quant = working_graph.add_propagating_quantizer([QuantizerConfig()],
                                                                     start_ip_node_key)
            ref_affected_edges = deepcopy(ref_prop_quant.affected_edges)
            ref_affected_edges.update(set(path))
            ref_affected_ip_nodes = deepcopy(ref_prop_quant.affected_ip_nodes)
            prop_quant = working_graph.propagate_quantizer_via_path(ref_prop_quant, path)
            final_node_key, _ = path[-1]
            for from_node_key, to_node_key in path:
                edge_data = working_graph.edges[from_node_key, to_node_key]
                assert edge_data[QPSG.AFFECTING_PROPAGATING_QUANTIZERS_ATTR] == [ref_prop_quant]
                to_node = working_graph.nodes[to_node_key]
                if to_node[QPSG.NODE_TYPE_NODE_ATTR] == QuantizerPropagationStateGraphNodeType.INSERTION_POINT:
                    assert to_node[QPSG.PROPAGATING_QUANTIZER_NODE_ATTR] is None
                from_node = working_graph.nodes[from_node_key]
                if from_node[QPSG.NODE_TYPE_NODE_ATTR] == QuantizerPropagationStateGraphNodeType.INSERTION_POINT:
                    ref_affected_ip_nodes.add(from_node_key)

            final_node_key, _ = path[-1]
            final_node = working_graph.nodes[final_node_key]
            assert final_node[QPSG.PROPAGATING_QUANTIZER_NODE_ATTR] == ref_prop_quant

            assert prop_quant.current_location_node_key == final_node_key
            assert prop_quant.propagation_path == path
            assert prop_quant.affected_edges == ref_affected_edges
            assert prop_quant.affected_ip_nodes == ref_affected_ip_nodes

    START_TARGET_ACCEPTING_NODES = [
        (InsertionPointGraph.get_pre_hook_node_key("H"),
         InsertionPointGraph.get_pre_hook_node_key("G"),
         InsertionPointGraph.get_post_hook_node_key("G")),

        (InsertionPointGraph.get_pre_hook_node_key("G"),
         InsertionPointGraph.get_post_hook_node_key("F"),
         InsertionPointGraph.get_post_hook_node_key("F")),

        (InsertionPointGraph.get_pre_hook_node_key("A"),
         None,
         None),

        (InsertionPointGraph.get_pre_hook_node_key("F"),
         InsertionPointGraph.get_pre_hook_node_key("C"),
         InsertionPointGraph.get_post_hook_node_key("C")),

        (InsertionPointGraph.get_pre_hook_node_key("D"),
         InsertionPointGraph.get_pre_hook_node_key("A"),
         InsertionPointGraph.get_post_hook_node_key("A")),
    ]

    @staticmethod
    @pytest.fixture(params=START_TARGET_ACCEPTING_NODES)
    def start_target_accepting_nodes(request):
        return request.param

    @pytest.mark.dependency(depends="propagate_via_path")
    def test_backtrack_propagation_until_accepting_location(self, start_target_accepting_nodes, mock_qp_graph):
        start_ip_node_key = start_target_accepting_nodes[0]
        target_ip_node_key = start_target_accepting_nodes[1]
        ref_last_accepting_location = start_target_accepting_nodes[2]

        prop_quant = mock_qp_graph.add_propagating_quantizer([QuantizerConfig()],
                                                             start_ip_node_key)
        ref_affected_edges = deepcopy(prop_quant.affected_edges)

        if target_ip_node_key is not None:
            # Here, the tested graph should have such a structure that there is only one path from target to start
            path = get_edge_paths_for_propagation(mock_qp_graph, target_ip_node_key, start_ip_node_key)[0]
            prop_quant = mock_qp_graph.propagate_quantizer_via_path(prop_quant, path)
            if ref_last_accepting_location is not None:
                resulting_path = get_edge_paths_for_propagation(mock_qp_graph,
                                                                ref_last_accepting_location, start_ip_node_key)[0]
                ref_affected_edges.update(set(resulting_path))

        assert prop_quant.last_accepting_location_node_key == ref_last_accepting_location
        prop_quant = mock_qp_graph.backtrack_propagation_until_accepting_location(prop_quant)

        if ref_last_accepting_location is None:
            assert prop_quant is None
        else:
            assert prop_quant.current_location_node_key == ref_last_accepting_location
            assert prop_quant.affected_edges == ref_affected_edges
            assert prop_quant.propagation_path == resulting_path

        if target_ip_node_key is not None and ref_last_accepting_location is not None:
            target_node = mock_qp_graph.nodes[target_ip_node_key]
            accepting_node = mock_qp_graph.nodes[ref_last_accepting_location]
            if ref_last_accepting_location != target_ip_node_key:
                assert target_node[QPSG.PROPAGATING_QUANTIZER_NODE_ATTR] is None
                assert target_ip_node_key not in prop_quant.affected_ip_nodes
            assert accepting_node[QPSG.PROPAGATING_QUANTIZER_NODE_ATTR] == prop_quant

    @pytest.mark.dependency(depends="propagate_via_path")
    def test_clone_propagating_quantizer(self, mock_qp_graph, start_target_nodes):
        start_ip_node_key = start_target_nodes[0]
        target_ip_node_key = start_target_nodes[1]

        # From "target" to "start" since propagation direction is inverse to edge direction
        # Only take one path out of possible paths for this test
        rev_path = get_edge_paths_for_propagation(mock_qp_graph, target_ip_node_key, start_ip_node_key)[0]

        ref_prop_quant = mock_qp_graph.add_propagating_quantizer([QuantizerConfig()],
                                                                 start_ip_node_key)

        prop_quant = mock_qp_graph.propagate_quantizer_via_path(ref_prop_quant, rev_path)

        cloned_prop_quant = mock_qp_graph.clone_propagating_quantizer(prop_quant)

        assert cloned_prop_quant.affected_ip_nodes == prop_quant.affected_ip_nodes
        assert cloned_prop_quant.affected_edges == prop_quant.affected_edges
        assert cloned_prop_quant.propagation_path == prop_quant.propagation_path
        assert cloned_prop_quant.current_location_node_key == prop_quant.current_location_node_key

        for ip_node_key in prop_quant.affected_ip_nodes:
            node = mock_qp_graph.nodes[ip_node_key]
            assert cloned_prop_quant in node[QPSG.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]
        for from_node_key, to_node_key in prop_quant.affected_edges:
            edge = mock_qp_graph.edges[from_node_key, to_node_key]
            assert cloned_prop_quant in edge[QPSG.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]

    START_TARGET_NODES_FOR_TWO_QUANTIZERS = [
        (InsertionPointGraph.get_pre_hook_node_key("E"),
         InsertionPointGraph.get_post_hook_node_key("C"),
         InsertionPointGraph.get_pre_hook_node_key("H"),
         InsertionPointGraph.get_post_hook_node_key("G")),

        (InsertionPointGraph.get_pre_hook_node_key("C"),
         InsertionPointGraph.get_post_hook_node_key("A"),
         InsertionPointGraph.get_pre_hook_node_key("H"),
         InsertionPointGraph.get_pre_hook_node_key("D")),

        # Simulated quantizer merging result
        (InsertionPointGraph.get_pre_hook_node_key("G"),
         InsertionPointGraph.get_pre_hook_node_key("E"),
         InsertionPointGraph.get_pre_hook_node_key("G"),
         InsertionPointGraph.get_post_hook_node_key("D"))
    ]

    @staticmethod
    @pytest.fixture(params=START_TARGET_NODES_FOR_TWO_QUANTIZERS)
    def start_target_nodes_for_two_quantizers(request):
        return request.param

    @pytest.mark.dependency(depends="propagate_via_path")
    def test_remove_propagating_quantizer(self, mock_qp_graph, start_target_nodes_for_two_quantizers):
        start_ip_node_key_remove = start_target_nodes_for_two_quantizers[0]
        target_ip_node_key_remove = start_target_nodes_for_two_quantizers[1]

        start_ip_node_key_keep = start_target_nodes_for_two_quantizers[2]
        target_ip_node_key_keep = start_target_nodes_for_two_quantizers[3]

        # From "target" to "start" since propagation direction is inverse to edge direction
        # Only take one path out of possible paths for this test
        rev_path_remove = get_edge_paths_for_propagation(mock_qp_graph,
                                                         target_ip_node_key_remove,
                                                         start_ip_node_key_remove)[0]
        rev_path_keep = get_edge_paths_for_propagation(mock_qp_graph,
                                                       target_ip_node_key_keep,
                                                       start_ip_node_key_keep)[0]

        prop_quant_to_remove = mock_qp_graph.add_propagating_quantizer([QuantizerConfig()],
                                                                       start_ip_node_key_remove)
        prop_quant_to_remove = mock_qp_graph.propagate_quantizer_via_path(prop_quant_to_remove, rev_path_remove)

        prop_quant_to_keep = mock_qp_graph.add_propagating_quantizer([QuantizerConfig()],
                                                                     start_ip_node_key_keep)

        prop_quant_to_keep = mock_qp_graph.propagate_quantizer_via_path(prop_quant_to_keep, rev_path_keep)

        affected_ip_nodes = deepcopy(prop_quant_to_remove.affected_ip_nodes)
        affected_edges = deepcopy(prop_quant_to_keep.affected_edges)
        last_location = prop_quant_to_remove.current_location_node_key
        ref_quant_to_keep_state_dict = deepcopy(prop_quant_to_keep.__dict__)

        mock_qp_graph.remove_propagating_quantizer(prop_quant_to_remove)

        last_node = mock_qp_graph.nodes[last_location]
        assert last_node[QPSG.PROPAGATING_QUANTIZER_NODE_ATTR] is None

        for ip_node_key in affected_ip_nodes:
            node = mock_qp_graph.nodes[ip_node_key]
            assert prop_quant_to_remove not in node[QPSG.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]

        for from_node_key, to_node_key in affected_edges:
            edge = mock_qp_graph.edges[from_node_key, to_node_key]
            assert prop_quant_to_remove not in edge[QPSG.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]

        assert prop_quant_to_keep.__dict__ == ref_quant_to_keep_state_dict

        for ip_node_key in prop_quant_to_keep.affected_ip_nodes:
            node = mock_qp_graph.nodes[ip_node_key]
            assert prop_quant_to_keep in node[
                QPSG.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]

        for from_node_key, to_node_key in prop_quant_to_keep.affected_edges:
            edge = mock_qp_graph.edges[from_node_key, to_node_key]
            assert prop_quant_to_keep in edge[
                QPSG.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]

    QUANTIZABLE_NODES_START_NODES_DOMINATED_NODES = [
        (["D", "E", "F"],
         {
             "B": {"D", "E"},
             InsertionPointGraph.get_pre_hook_node_key("B"): {"D", "E"},
             "E": {"F"},
             InsertionPointGraph.get_post_hook_node_key("D"): {"F"},
             "A": {"D", "E"},
             InsertionPointGraph.get_pre_hook_node_key("G"): set()
         }),
        (["C", "E", "H"],
         {
             InsertionPointGraph.get_pre_hook_node_key("C"): {"C"},
             InsertionPointGraph.get_post_hook_node_key("C"): {"E"},
             "D": {"H"},
             InsertionPointGraph.get_pre_hook_node_key("B"): {"C", "H"}, # corner case - has a branch without quantizers
             InsertionPointGraph.get_post_hook_node_key("H"): set()
         })
    ]

    @staticmethod
    @pytest.fixture(params=QUANTIZABLE_NODES_START_NODES_DOMINATED_NODES)
    def dominated_nodes_test_struct(request):
        return request.param

    def test_get_quantizable_op_nodes_immediately_dominated_by_node(self, mock_qp_graph, dominated_nodes_test_struct):
        nodes_to_mark_as_quantizable = dominated_nodes_test_struct[0]

        for node in mock_qp_graph.nodes.values():
            node[QPSG.QUANTIZATION_TRAIT_NODE_ATTR] = QuantizationTrait.QUANTIZATION_AGNOSTIC

        traits_to_mark_with = [QuantizationTrait.INPUTS_QUANTIZABLE,
                               QuantizationTrait.NON_QUANTIZABLE]

        for trait in traits_to_mark_with:
            for node_key in nodes_to_mark_as_quantizable:
                node = mock_qp_graph.nodes[node_key]
                node[QPSG.QUANTIZATION_TRAIT_NODE_ATTR] = trait

            for start_node_key, ref_dominated_quantizable_nodes_set in dominated_nodes_test_struct[1].items():
                dominated_quantizable_nodes_list = mock_qp_graph.get_quantizable_op_nodes_immediately_dominated_by_node(
                    start_node_key)
                assert set(dominated_quantizable_nodes_list) == ref_dominated_quantizable_nodes_set

    @staticmethod
    def get_model_graph():
        mock_node_attrs = get_mock_nncf_node_attrs()
        mock_graph = nx.DiGraph()

        #     (A)
        #      |
        #     (B)
        #   /     \
        # (C)     (D)
        #  |       |
        # (F)     (E)
        #
        #

        node_keys = ['A', 'B', 'C', 'D', 'E', 'F']
        for node_key in node_keys:
            mock_graph.add_node(node_key, **mock_node_attrs)

        mock_graph.add_edges_from([('A', 'B'), ('B', 'C'), ('B', 'D'), ('D', 'E'), ('C', 'F')])
        return mock_graph



    StateQuantizerTestStruct = namedtuple('StateQuantizerTestStruct',
                                          ('init_node_to_trait_and_configs_dict',
                                           'starting_quantizer_ip_node',
                                           'target_node_for_quantizer',
                                           'is_merged',
                                           'prop_path'))

    SetQuantizersTestStruct = namedtuple('SetQuantizersTestStruct',
                                         ('start_set_quantizers',
                                          'expected_set_quantizers'))

    MERGE_QUANTIZER_INTO_PATH_TEST_CASES = [
        SetQuantizersTestStruct(
            start_set_quantizers=[
                StateQuantizerTestStruct(
                    init_node_to_trait_and_configs_dict=
                    {
                        'E': (QuantizationTrait.INPUTS_QUANTIZABLE,
                              [QuantizerConfig()])
                    },
                    starting_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('E'),
                    target_node_for_quantizer=InsertionPointGraph.get_pre_hook_node_key('B'),
                    is_merged=False,
                    prop_path=None

                ),
                StateQuantizerTestStruct(
                    init_node_to_trait_and_configs_dict=
                    {
                        'F': (QuantizationTrait.INPUTS_QUANTIZABLE,
                              [QuantizerConfig()]),
                    },
                    starting_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('F'),
                    target_node_for_quantizer=InsertionPointGraph.get_pre_hook_node_key('C'),
                    is_merged=True,
                    prop_path=[(InsertionPointGraph.get_post_hook_node_key('B'),
                                InsertionPointGraph.get_pre_hook_node_key('C'))]
                )

            ],
            expected_set_quantizers=[
                StateQuantizerTestStruct(
                    init_node_to_trait_and_configs_dict=
                    {
                        'PRE HOOK B': (QuantizationTrait.INPUTS_QUANTIZABLE,
                                       [QuantizerConfig()])

                    },
                    starting_quantizer_ip_node=['E', 'F'],
                    target_node_for_quantizer=InsertionPointGraph.get_pre_hook_node_key('B'),
                    is_merged=False,
                    prop_path=None
                )
            ]
        ),
        SetQuantizersTestStruct(
            start_set_quantizers=[
                StateQuantizerTestStruct(
                    init_node_to_trait_and_configs_dict=
                    {
                        'E': (QuantizationTrait.INPUTS_QUANTIZABLE,
                              [QuantizerConfig()])
                    },
                    starting_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('E'),
                    target_node_for_quantizer=InsertionPointGraph.get_pre_hook_node_key('B'),
                    is_merged=False,
                    prop_path=None

                ),
                StateQuantizerTestStruct(
                    init_node_to_trait_and_configs_dict=
                    {
                        'F': (QuantizationTrait.INPUTS_QUANTIZABLE,
                              [QuantizerConfig()]),
                    },
                    starting_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('F'),
                    target_node_for_quantizer=InsertionPointGraph.get_post_hook_node_key('B'),
                    is_merged=True,
                    prop_path=[('B', InsertionPointGraph.get_post_hook_node_key('B'))]
                )

            ],
            expected_set_quantizers=[
                StateQuantizerTestStruct(
                    init_node_to_trait_and_configs_dict=
                    {
                        'B': (QuantizationTrait.INPUTS_QUANTIZABLE,
                              [QuantizerConfig()])

                    },
                    starting_quantizer_ip_node=['E', 'F'],
                    target_node_for_quantizer=InsertionPointGraph.get_pre_hook_node_key('B'),
                    is_merged=False,
                    prop_path=None
                )
            ]
        ),
        SetQuantizersTestStruct(
            start_set_quantizers=[
                StateQuantizerTestStruct(
                    init_node_to_trait_and_configs_dict=
                    {
                        'E': (QuantizationTrait.INPUTS_QUANTIZABLE,
                              [QuantizerConfig()])
                    },
                    starting_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('E'),
                    target_node_for_quantizer=InsertionPointGraph.get_post_hook_node_key('B'),
                    is_merged=False,
                    prop_path=None

                ),
                StateQuantizerTestStruct(
                    init_node_to_trait_and_configs_dict=
                    {
                        'F': (QuantizationTrait.INPUTS_QUANTIZABLE,
                              [QuantizerConfig()]),
                    },
                    starting_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('F'),
                    target_node_for_quantizer=InsertionPointGraph.get_pre_hook_node_key('C'),
                    is_merged=True,
                    prop_path=[(InsertionPointGraph.get_post_hook_node_key('B'),
                                InsertionPointGraph.get_pre_hook_node_key('C'))]
                )

            ],
            expected_set_quantizers=[
                StateQuantizerTestStruct(
                    init_node_to_trait_and_configs_dict=
                    {
                        'B': (QuantizationTrait.INPUTS_QUANTIZABLE,
                              [QuantizerConfig()])

                    },
                    starting_quantizer_ip_node=['E', 'F'],
                    target_node_for_quantizer=InsertionPointGraph.get_post_hook_node_key('B'),
                    is_merged=False,
                    prop_path=None
                )
            ]
        )
    ]

    @staticmethod
    @pytest.fixture(params=MERGE_QUANTIZER_INTO_PATH_TEST_CASES)
    def merge_quantizer_into_path_test_struct(request):
        return request.param

    def test_merge_quantizer_into_path(self, merge_quantizer_into_path_test_struct):

        mock_graph = self.get_model_graph()
        ip_graph = InsertionPointGraph(mock_graph)
        quant_prop_graph = QPSG(ip_graph)

        for quantizers_test_struct in merge_quantizer_into_path_test_struct.start_set_quantizers:

            init_node_to_trait_and_configs_dict = quantizers_test_struct.init_node_to_trait_and_configs_dict
            starting_quantizer_ip_node = quantizers_test_struct.starting_quantizer_ip_node
            target_node = quantizers_test_struct.target_node_for_quantizer
            is_merged = quantizers_test_struct.is_merged
            prop_path = quantizers_test_struct.prop_path
            for node in quant_prop_graph.nodes.values():
                node[QPSG.QUANTIZATION_TRAIT_NODE_ATTR] = QuantizationTrait.QUANTIZATION_AGNOSTIC
            master_prop_quant = None
            merged_prop_quant = []
            for node_key, trait_and_configs_tuple in init_node_to_trait_and_configs_dict.items():
                trait = trait_and_configs_tuple[0]
                qconfigs = trait_and_configs_tuple[1]
                quant_prop_graph.nodes[node_key][QPSG.QUANTIZATION_TRAIT_NODE_ATTR] = trait
                if trait == QuantizationTrait.INPUTS_QUANTIZABLE:
                    ip_node_key = InsertionPointGraph.get_pre_hook_node_key(node_key)
                    prop_quant = quant_prop_graph.add_propagating_quantizer(qconfigs,
                                                                            ip_node_key)
                    if ip_node_key == starting_quantizer_ip_node:
                        master_prop_quant = prop_quant

            path = get_edge_paths_for_propagation(quant_prop_graph,
                                                  target_node,
                                                  starting_quantizer_ip_node)
            master_prop_quant = quant_prop_graph.propagate_quantizer_via_path(master_prop_quant,
                                                                              path[0])
            if is_merged:
                merged_prop_quant.append((master_prop_quant, prop_path))

        for prop_quant, prop_path in merged_prop_quant:
            quant_prop_graph.merge_quantizer_into_path(prop_quant, prop_path)

        expected_quantizers_test_struct = merge_quantizer_into_path_test_struct.expected_set_quantizers
        self.check_final_state_qpsg(quant_prop_graph, expected_quantizers_test_struct)


    @staticmethod
    def check_final_state_qpsg(final_quant_prop_graph, expected_quantizers_test_struct):
        for quantizer_param in expected_quantizers_test_struct:
            from_node_key = quantizer_param.target_node_for_quantizer
            expected_prop_path = set()
            target_node = quantizer_param.target_node_for_quantizer
            for start_node in quantizer_param.starting_quantizer_ip_node:
                added_path = get_edge_paths_for_propagation(final_quant_prop_graph,
                                                            target_node,
                                                            start_node)
                expected_prop_path.update(added_path[0])

            quantizer = final_quant_prop_graph.nodes[from_node_key][QPSG.PROPAGATING_QUANTIZER_NODE_ATTR]

            assert quantizer is not None

            for from_node_key, to_node_key in expected_prop_path:
                assert quantizer in final_quant_prop_graph.edges[(from_node_key, to_node_key)][
                    QPSG.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]
                from_node = final_quant_prop_graph.nodes[from_node_key]
                from_node_type = from_node[QPSG.NODE_TYPE_NODE_ATTR]
                if from_node_type == QuantizerPropagationStateGraphNodeType.INSERTION_POINT:
                    # pylint:disable=line-too-long
                    assert quantizer in final_quant_prop_graph.nodes[from_node_key][QPSG.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]

            assert quantizer.affected_edges == expected_prop_path
