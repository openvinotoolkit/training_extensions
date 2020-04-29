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
from collections import Counter

from nncf.dynamic_graph.graph import NNCFGraph, NNCFGraphPatternIO, NNCFGraphEdge, NNCFNode


def test_graph_pattern_io_building():
    graph = NNCFGraph()
    #   1
    # /   \
    # 2   |
    # |   |
    # 3   |
    # \   /
    #   4
    # / | \
    # 5 6 7
    # |
    # 8

    #pylint:disable=protected-access
    node_keys = ['1', '2', '3', '4', '5', '6', '7', '8']
    for idx, node_key in enumerate(node_keys):
        attrs = {
            NNCFGraph.ID_NODE_ATTR: idx + 1,
            NNCFGraph.KEY_NODE_ATTR: node_key,
            NNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR: None,
        }
        graph._nx_graph.add_node(node_key, **attrs)

    edge_attr = {NNCFGraph.ACTIVATION_SHAPE_EDGE_ATTR: None}
    graph._nx_graph.add_edges_from([('1', '2'), ('1', '4'), ('2', '3'), ('3', '4'), ('4', '5'),
                                    ('4', '6'), ('4', '7'), ('5', '8')], **edge_attr)
    graph._node_id_to_key_dict.update({k + 1: v for k, v in enumerate(node_keys)})

    def make_mock_edge(from_id: int, to_id: int):
        return NNCFGraphEdge(NNCFNode(from_id, None),
                             NNCFNode(to_id, None), None)

    def make_mock_node(id_: int):
        return NNCFNode(id_, None)

    ref_patterns_and_ios = [
        (['1', '2'], NNCFGraphPatternIO(input_edges=[],
                                        input_nodes=[make_mock_node(1)],
                                        output_edges=[make_mock_edge(2, 3),
                                                      make_mock_edge(1, 4)],
                                        output_nodes=[])),
        (['3'], NNCFGraphPatternIO(input_edges=[make_mock_edge(2, 3)],
                                   input_nodes=[],
                                   output_edges=[make_mock_edge(3, 4)],
                                   output_nodes=[])),
        (['1', '2', '3'], NNCFGraphPatternIO(input_edges=[],
                                             input_nodes=[make_mock_node(1)],
                                             output_edges=[make_mock_edge(3, 4),
                                                           make_mock_edge(1, 4)],
                                             output_nodes=[])),
        (['4'], NNCFGraphPatternIO(input_edges=[make_mock_edge(3, 4),
                                                make_mock_edge(1, 4)],
                                   input_nodes=[],
                                   output_edges=[make_mock_edge(4, 5),
                                                 make_mock_edge(4, 6),
                                                 make_mock_edge(4, 7)],
                                   output_nodes=[])),
        (['5', '6', '8'], NNCFGraphPatternIO(input_edges=[make_mock_edge(4, 5),
                                                          make_mock_edge(4, 6)],
                                             input_nodes=[],
                                             output_edges=[],
                                             output_nodes=[make_mock_node(6),
                                                           make_mock_node(8)])),
        (['7'], NNCFGraphPatternIO(input_edges=[make_mock_edge(4, 7)],
                                   input_nodes=[],
                                   output_edges=[],
                                   output_nodes=[make_mock_node(7)]))
    ]

    for pattern, ref_pattern_io in ref_patterns_and_ios:
        test_pattern_io = graph._get_nncf_graph_pattern_io_list(pattern)
        assert Counter(test_pattern_io.input_edges) == Counter(ref_pattern_io.input_edges)
        assert Counter(test_pattern_io.output_edges) == Counter(ref_pattern_io.output_edges)
        assert Counter(test_pattern_io.input_nodes) == Counter(ref_pattern_io.input_nodes)
        assert Counter(test_pattern_io.output_nodes) == Counter(ref_pattern_io.output_nodes)
