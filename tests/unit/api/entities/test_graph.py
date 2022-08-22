# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest
from networkx.classes.reportviews import (
    DegreeView,
    DiMultiDegreeView,
    EdgeDataView,
    EdgeView,
    InMultiDegreeView,
    NodeView,
    OutMultiDegreeView,
    OutMultiEdgeDataView,
    OutMultiEdgeView,
)
from networkx.exception import NetworkXError, NetworkXNotImplemented

from otx.api.entities.graph import Graph, MultiDiGraph
from tests.unit.api.constants.components import OtxSdkComponent
from tests.unit.api.constants.requirements import Requirements


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestGraph:
    @staticmethod
    def add_edges_to_graph(graph_to_add_edges: Graph, edges_parameters: list) -> None:
        for edge_dict in edges_parameters:
            graph_to_add_edges.add_edge(**edge_dict)

    def non_directed_graph(self) -> Graph:
        non_directed_graph = Graph()
        self.add_edges_to_graph(
            non_directed_graph,
            [
                {"node1": 1, "node2": 2, "edge_value": 2},
                {"node1": 2, "node2": 3, "edge_value": 3},
                {"node1": 3, "node2": 4, "edge_value": 4},
            ],
        )
        return non_directed_graph

    def directed_graph(self) -> Graph:
        directed_graph = Graph(directed=True)
        self.add_edges_to_graph(
            directed_graph,
            [
                {"node1": "A", "node2": "B", "edge_value": 1},
                {"node1": "B", "node2": "C", "edge_value": 2},
                {"node1": "C", "node2": "D", "edge_value": 3},
                {"node1": "D", "node2": "A", "edge_value": 5},
                {"node1": "A", "node2": "D", "edge_value": 4},
            ],
        )
        return directed_graph

    @staticmethod
    def check_graph_non_list_attributes(expected_attributes_dicts: list) -> None:
        for expected_attribute_dict in expected_attributes_dicts:
            assert expected_attribute_dict.get("attribute") == expected_attribute_dict.get("expected_value")

    @staticmethod
    def check_graph_list_attributes(actual_expected_attributes_dict: list) -> None:
        for expected_attribute_dict in actual_expected_attributes_dict:
            attribute = expected_attribute_dict.get("attribute")
            assert isinstance(attribute, expected_attribute_dict.get("expected_type"))
            assert list(attribute) == expected_attribute_dict.get("expected_value")

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
    def test_graph_initialization(self):
        """
        <b>Description:</b>
        Check Graph class object initialization

        <b>Input data:</b>
        Graph objects with specified "directed" and "edges" parameters

        <b>Expected results:</b>
        Test passes if Graph object "directed" attribute, "edges", "nodes" and "num_labels" properties and "num_nodes"
        method return expected values

        <b>Steps</b>
        1. Check initialization of non-directed Graph object
        2. Check initialization of directed Graph object
        """
        # Checking not directed Graph values
        non_directed_graph = self.non_directed_graph()
        self.check_graph_non_list_attributes(
            [
                {"attribute": non_directed_graph.directed, "expected_value": False},
                {"attribute": non_directed_graph.num_labels, "expected_value": 4},
                {"attribute": non_directed_graph.num_nodes(), "expected_value": 4},
            ]
        )
        self.check_graph_list_attributes(
            [
                {
                    "attribute": non_directed_graph.edges,
                    "expected_type": EdgeDataView,
                    "expected_value": [
                        (1, 2, {"value": 2}),
                        (2, 3, {"value": 3}),
                        (3, 4, {"value": 4}),
                    ],
                },
                {
                    "attribute": non_directed_graph.nodes,
                    "expected_type": NodeView,
                    "expected_value": [1, 2, 3, 4],
                },
            ]
        )
        # Checking directed Graph values
        directed_graph = self.directed_graph()
        self.check_graph_non_list_attributes(
            [
                {"attribute": directed_graph.directed, "expected_value": True},
                {"attribute": directed_graph.num_labels, "expected_value": 4},
                {"attribute": directed_graph.num_nodes(), "expected_value": 4},
            ]
        )
        self.check_graph_list_attributes(
            [
                {
                    "attribute": directed_graph.edges,
                    "expected_type": OutMultiEdgeDataView,
                    "expected_value": [
                        ("A", "B", 0, {"value": 1}),
                        ("A", "D", 0, {"value": 4}),
                        ("B", "C", 0, {"value": 2}),
                        ("C", "D", 0, {"value": 3}),
                        ("D", "A", 0, {"value": 5}),
                    ],
                },
                {
                    "attribute": directed_graph.nodes,
                    "expected_type": NodeView,
                    "expected_value": ["A", "B", "C", "D"],
                },
            ]
        )

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_graph_get_graph(self):
        """
        <b>Description:</b>
        Check get_graph method of Graph class object

        <b>Input data:</b>
        Graph objects with specified "directed" and "edges" parameters

        <b>Expected results:</b>
        Test passes if Graph object returned by get_graph method returns expected attributes

        <b>Steps</b>
        1. Check attributes of Graph returned by get_graph method for non-directed Graph object
        2. Check attributes of Graph returned by get_graph method for directed Graph object
        """
        # Checking non-directed Graph attributes
        non_directed_graph = self.non_directed_graph().get_graph()
        self.check_graph_list_attributes(
            [
                {
                    "attribute": non_directed_graph.degree,
                    "expected_type": DegreeView,
                    "expected_value": [(1, 1), (2, 2), (3, 2), (4, 1)],
                },
                {
                    "attribute": non_directed_graph.edges,
                    "expected_type": EdgeView,
                    "expected_value": [(1, 2), (2, 3), (3, 4)],
                },
                {
                    "attribute": non_directed_graph.nodes,
                    "expected_type": NodeView,
                    "expected_value": [1, 2, 3, 4],
                },
            ]
        )
        # Checking directed Graph attributes
        directed_graph = self.directed_graph().get_graph()
        self.check_graph_list_attributes(
            [
                {
                    "attribute": directed_graph.nodes,
                    "expected_type": NodeView,
                    "expected_value": ["A", "B", "C", "D"],
                },
                {
                    "attribute": directed_graph.edges,
                    "expected_type": OutMultiEdgeView,
                    "expected_value": [
                        ("A", "B", 0),
                        ("A", "D", 0),
                        ("B", "C", 0),
                        ("C", "D", 0),
                        ("D", "A", 0),
                    ],
                },
                {
                    "attribute": directed_graph.degree,
                    "expected_type": DiMultiDegreeView,
                    "expected_value": [("A", 3), ("B", 2), ("C", 2), ("D", 3)],
                },
                {
                    "attribute": directed_graph.in_degree,
                    "expected_type": InMultiDegreeView,
                    "expected_value": [("A", 1), ("B", 1), ("C", 1), ("D", 2)],
                },
                {
                    "attribute": directed_graph.out_degree,
                    "expected_type": OutMultiDegreeView,
                    "expected_value": [("A", 2), ("B", 1), ("C", 1), ("D", 1)],
                },
            ]
        )

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_graph_set_graph(self):
        """
        <b>Description:</b>
        Check set_graph method of Graph class object

        <b>Input data:</b>
        Graph objects with specified "directed" and "edges" parameters

        <b>Expected results:</b>
        Test passes if graph attribute returned after set_graph method has expected values

        <b>Steps</b>
        1. Check graph attribute after setting unequal non-directed graph to non-directed Graph object
        2. Check graph attribute after setting directed graph to non-directed Graph object
        3. Check graph attribute after setting unequal directed graph to directed Graph object
        4. Check graph attribute after setting non-directed graph to directed Graph object
        """
        # Setting unequal non-directed graph to non-directed Graph
        non_directed_graph = self.non_directed_graph()
        unequal_non_directed_graph = Graph()
        self.add_edges_to_graph(
            unequal_non_directed_graph,
            [
                {"node1": 1, "node2": 3, "edge_value": 2},
                {"node1": 2, "node2": 4, "edge_value": 2},
            ],
        )
        non_directed_graph.set_graph(unequal_non_directed_graph)
        assert non_directed_graph.get_graph() == unequal_non_directed_graph
        # Setting directed graph to non-directed Graph
        non_directed_graph = self.non_directed_graph()
        directed_graph = self.directed_graph()
        non_directed_graph.set_graph(directed_graph)
        assert non_directed_graph.get_graph() == directed_graph
        # Setting unequal directed graph to directed Graph
        unequal_directed_graph = Graph(directed=True)
        self.add_edges_to_graph(
            unequal_directed_graph,
            [
                {"node1": "A", "node2": "B", "edge_value": 1},
                {"node1": "B", "node2": "C", "edge_value": 2},
                {"node1": "C", "node2": "B", "edge_value": 1},
            ],
        )
        directed_graph.set_graph(unequal_directed_graph)
        assert directed_graph.get_graph() == unequal_directed_graph
        # Setting non-directed graph to directed Graph
        directed_graph = self.directed_graph()
        non_directed_graph = self.non_directed_graph()
        directed_graph.set_graph(non_directed_graph)
        assert directed_graph.get_graph() == non_directed_graph

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_graph_add_edge(self):
        """
        <b>Description:</b>
        Check add_edge method of Graph class object

        <b>Input data:</b>
        Graph objects with specified "directed" and "edges" parameters

        <b>Expected results:</b>
        Test passes if edges attribute returned after add_edge method has expected values

        <b>Steps</b>
        1. Check add_edge method by adding equal edge to non-directed graph (expected, that edges not changed)
        2. Check add_edge method by adding reversed edge to non-directed graph (expected, that edges not changed)
        3. Check add_edge method by adding unequal edge to non-directed graph
        4. Check add_edge method by adding equal edge to directed graph
        5. Check add_edge method by adding reversed edge to directed graph
        6. Check add_edge method by adding unequal edge to directed graph
        """
        # Scenario for non-directed graph
        non_directed_graph = self.non_directed_graph()
        non_directed_graph_edges = [
            (1, 2, {"value": 2}),
            (2, 3, {"value": 3}),
            (3, 4, {"value": 4}),
        ]
        # Adding equal edge to non-directed graph
        non_directed_graph.add_edge(node1=2, node2=3, edge_value=3)
        assert list(non_directed_graph.edges) == non_directed_graph_edges
        # Adding equal reversed edge to non-directed graph
        non_directed_graph.add_edge(node1=3, node2=2, edge_value=3)
        assert list(non_directed_graph.edges) == non_directed_graph_edges
        # Adding unequal edge with existing nodes to non-directed graph
        non_directed_graph.add_edge(node1=1, node2=3, edge_value=1)
        # Adding unequal edge with new node to non-directed graph
        non_directed_graph.add_edge(node1=4, node2=5, edge_value=2)
        assert list(non_directed_graph.edges) == [
            (1, 2, {"value": 2}),
            (1, 3, {"value": 1}),
            (2, 3, {"value": 3}),
            (3, 4, {"value": 4}),
            (4, 5, {"value": 2}),
        ]
        # Scenario for directed graph
        directed_graph = self.directed_graph()
        # Adding equal edge to directed graph
        directed_graph.add_edge(node1="C", node2="D", edge_value=3)
        assert list(directed_graph.edges) == [
            ("A", "B", 0, {"value": 1}),
            ("A", "D", 0, {"value": 4}),
            ("B", "C", 0, {"value": 2}),
            ("C", "D", 0, {"value": 3}),
            ("C", "D", 1, {"value": 3}),
            ("D", "A", 0, {"value": 5}),
        ]
        # Adding equal reversed edge to directed graph
        directed_graph.add_edge(node1="B", node2="A", edge_value=2)
        assert list(directed_graph.edges) == [
            ("A", "B", 0, {"value": 1}),
            ("A", "D", 0, {"value": 4}),
            ("B", "C", 0, {"value": 2}),
            ("B", "A", 0, {"value": 2}),
            ("C", "D", 0, {"value": 3}),
            ("C", "D", 1, {"value": 3}),
            ("D", "A", 0, {"value": 5}),
        ]
        # Adding unequal edge with new nodes to non-directed graph
        directed_graph.add_edge(node1="E", node2="F")
        assert list(directed_graph.edges) == [
            ("A", "B", 0, {"value": 1}),
            ("A", "D", 0, {"value": 4}),
            ("B", "C", 0, {"value": 2}),
            ("B", "A", 0, {"value": 2}),
            ("C", "D", 0, {"value": 3}),
            ("C", "D", 1, {"value": 3}),
            ("D", "A", 0, {"value": 5}),
            ("E", "F", 0, {"value": None}),
        ]

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_graph_has_edge_between(self):
        """
        <b>Description:</b>
        Check has_edge_between method of Graph class object

        <b>Input data:</b>
        Graph objects with specified "directed" and "edges" parameters

        <b>Expected results:</b>
        Test passes if has_edge_between method returns expected value

        <b>Steps</b>
        1. Check value returned by has_edge_between method for non-directed graph
        2. Check value returned by has_edge_between method for directed graph
        """
        # Scenario for non-directed graph
        non_directed_graph = self.non_directed_graph()
        for node_1, node_2 in ([1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3]):
            assert non_directed_graph.has_edge_between(node_1, node_2)
        for node_1, node_2 in ([1, 3], [3, 1], [1, 4], [4, 1], [2, 4], [4, 2]):
            assert not non_directed_graph.has_edge_between(node_1, node_2)
        # Scenario for directed graph
        directed_graph = self.directed_graph()
        for node_1, node_2 in (
            ["B", "A"],
            ["C", "B"],
            ["D", "C"],
            ["A", "D"],
            ["D", "A"],
        ):
            assert directed_graph.has_edge_between(node_1, node_2)
        for node_1, node_2 in (
            ["A", "B"],
            ["B", "C"],
            ["C", "D"],
            ["A", "C"],
            ["C", "A"],
            ["B", "D"],
            ["D", "B"],
        ):
            assert not directed_graph.has_edge_between(node_1, node_2)

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_graph_neighbors(self):
        """
        <b>Description:</b>
        Check neighbors method of Graph class object

        <b>Input data:</b>
        Graph objects with specified "directed" and "edges" parameters

        <b>Expected results:</b>
        Test passes if neighbors method returns expected value

        <b>Steps</b>
        1. Check value returned by neighbors method for non-directed graph
        2. Check value returned by neighbors method for directed graph
        """
        # Scenario for non-directed graph
        non_directed_graph = self.non_directed_graph()
        for node, expected_neighbors in ([1, [2]], [2, [1, 3]], [3, [2, 4]], [4, [3]]):
            assert non_directed_graph.neighbors(node) == expected_neighbors
        assert non_directed_graph.neighbors(5) == []
        # Scenario for directed graph
        directed_graph = self.directed_graph()
        for node, expected_neighbors in (
            ["A", ["B", "D"]],
            ["B", ["C"]],
            ["C", ["D"]],
            ["D", ["A"]],
        ):
            assert directed_graph.neighbors(node) == expected_neighbors
        assert directed_graph.neighbors(1) == []

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_graph_find_out_edges(self):
        """
        <b>Description:</b>
        Check find_out_edges method of Graph class object

        <b>Input data:</b>
        Graph objects with specified "directed" and "edges" parameters

        <b>Expected results:</b>
        Test passes if find_out_edges method returns expected value

        <b>Steps</b>
        1. Check empty list returned by find_out_edges method for non-directed graph
        2. Check value returned by find_out_edges method for directed graph
        """
        # Scenario for non-directed graph
        non_directed_graph = self.non_directed_graph()
        for node in [1, 2, 3, 4]:
            assert non_directed_graph.find_out_edges(node) == []
        # Scenario for directed graph
        directed_graph = self.directed_graph()
        for node, expected_edges in (
            ["A", [("A", "B"), ("A", "D")]],
            ["B", [("B", "C")]],
            ["C", [("C", "D")]],
            ["D", [("D", "A")]],
        ):
            assert list(directed_graph.find_out_edges(node)) == expected_edges
        with pytest.raises(KeyError):
            directed_graph.find_out_edges(1)

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_graph_find_in_edges(self):
        """
        <b>Description:</b>
        Check find_in_edges method of Graph class object

        <b>Input data:</b>
        Graph objects with specified "directed" and "edges" parameters

        <b>Expected results:</b>
        Test passes if find_in_edges method returns expected value

        <b>Steps</b>
        1. Check empty list returned by find_in_edges method for non-directed graph
        2. Check value returned by find_in_edges method for directed graph
        """
        # Scenario for non-directed graph
        non_directed_graph = self.non_directed_graph()
        for node in [1, 2, 3, 4]:
            assert non_directed_graph.find_in_edges(node) == []
        # Scenario for directed graph
        directed_graph = self.directed_graph()
        for node, expected_edges in (
            ["A", [("D", "A")]],
            ["B", [("A", "B")]],
            ["C", [("B", "C")]],
            ["D", [("C", "D"), ("A", "D")]],
        ):
            assert list(directed_graph.find_in_edges(node)) == expected_edges
        with pytest.raises(KeyError):
            directed_graph.find_in_edges(1)

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_graph_find_cliques(self):
        """
        <b>Description:</b>
        Check find_cliques method of Graph class object

        <b>Input data:</b>
        Graph objects with specified "directed" and "edges" parameters

        <b>Expected results:</b>
        Test passes if find_cliques method returns generator object with expected values

        <b>Steps</b>
        1. Check value returned by find_cliques generator for non-directed graph
        2. Check NetworkXNotImplemented exception raised when find_cliques method used for directed graph
        """
        # Scenario for non-directed graph
        non_directed_graph_cliques = self.non_directed_graph().find_cliques()
        for expected_clique in ([2, 1], [2, 3], [4, 3]):
            assert next(non_directed_graph_cliques) == expected_clique
        # Scenario for directed graph
        with pytest.raises(NetworkXNotImplemented):
            # this is for networkx<2.6.0
            cliques = self.directed_graph().find_cliques()
            if next(cliques, None):
                # this is for networkx>=2.6.0
                raise NetworkXNotImplemented

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
    def test_graph_num_labels(self):
        """
        <b>Description:</b>
        Check num_labels property of Graph class object

        <b>Input data:</b>
        Graph objects with specified "directed" and "edges" parameters

        <b>Expected results:</b>
        Test passes if num_labels property returns expected value

        <b>Steps</b>
        1. Check value returned by num_labels property for non-directed graph
        2. Check value returned by num_labels property for directed graph
        """
        for empty_graph in (Graph(), Graph(directed=True)):
            assert empty_graph.num_labels == 0
        # Scenario for non-directed graph
        non_directed_graph = self.non_directed_graph()
        assert non_directed_graph.num_labels == 4
        # Checking num_labels property after adding edge with non existing nodes to non-directed graph
        non_directed_graph.add_edge(5, 6)
        assert non_directed_graph.num_labels == 6
        # Checking num_labels property after removing edge (nodes not removed) from non-directed graph
        non_directed_graph.remove_edges(5, 6)
        assert non_directed_graph.num_labels == 6
        # Checking num_labels property after removing node from non-directed graph
        non_directed_graph.remove_node(1)
        assert non_directed_graph.num_labels == 5
        # Scenario for directed graph
        directed_graph = self.directed_graph()
        assert directed_graph.num_labels == 4
        # Checking num_labels property after adding edge with one existing node to directed graph
        directed_graph.add_edge(node1="A", node2="E", edge_value=7)
        assert directed_graph.num_labels == 5
        # Checking num_labels property after removing edge (nodes not removed) from non-directed graph
        directed_graph.remove_edges(node1="A", node2="B")
        assert directed_graph.num_labels == 5
        # Checking num_labels property after removing node from non-directed graph
        directed_graph.remove_node("E")
        assert directed_graph.num_labels == 4

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_graph_remove_edges(self):
        """
        <b>Description:</b>
        Check remove_edges method of Graph class object

        <b>Input data:</b>
        Graph objects with specified "directed" and "edges" parameters

        <b>Expected results:</b>
        Test passes if returned expected value of edges property after using remove_edges method on Graph class object

        <b>Steps</b>
        1. Check value returned by edges property after using remove_edges method by non-directed Graph class object
        2. Check value returned by edges property after using remove_edges method by directed Graph class object
        """
        # Scenario for non-directed graph
        non_directed_graph = self.non_directed_graph()
        # Removing two existing edges from non-directed graph
        non_directed_graph.remove_edges(1, 2)
        non_directed_graph.remove_edges(3, 2)
        assert list(non_directed_graph.edges) == [(3, 4, {"value": 4})]
        # Checking that NetworkXError exception raised when trying to remove non existing edges from non-directed graph
        for node_1, node_2 in ([1, 2], [1, 3], [5, 6]):
            with pytest.raises(NetworkXError):
                non_directed_graph.remove_edges(node_1, node_2)
        # Checking "edges" property of non-directed graph after adding edge which were removed in previous scenario
        non_directed_graph.add_edge(node1=1, node2=2, edge_value=2)
        assert list(non_directed_graph.edges) == [
            (1, 2, {"value": 2}),
            (3, 4, {"value": 4}),
        ]
        # Scenario for directed graph
        directed_graph = self.directed_graph()
        # Removing two existing edges from directed graph, edge "D"-"A" - multi directional
        directed_graph.remove_edges("A", "B")
        directed_graph.remove_edges("D", "A")
        assert list(directed_graph.edges) == [
            ("A", "D", 0, {"value": 4}),
            ("B", "C", 0, {"value": 2}),
            ("C", "D", 0, {"value": 3}),
        ]
        # Checking that NetworkXError exception raised when trying to remove non existing edges from directed graph
        for node_1, node_2 in (["A", "B"], ["A", "C"], ["E", "F"]):
            with pytest.raises(NetworkXError):
                non_directed_graph.remove_edges(node_1, node_2)
        # Checking "edges" property of directed graph after adding edge which were removed in previous scenario
        directed_graph.add_edge(node1="D", node2="A", edge_value=5)
        assert list(directed_graph.edges) == [
            ("A", "D", 0, {"value": 4}),
            ("B", "C", 0, {"value": 2}),
            ("C", "D", 0, {"value": 3}),
            ("D", "A", 0, {"value": 5}),
        ]

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_graph_remove_node(self):
        """
        <b>Description:</b>
        Check remove_node method of Graph class object

        <b>Input data:</b>
        Graph objects with specified "directed" and "edges" parameters

        <b>Expected results:</b>
        Test passes if returned expected values of nodes property and num_nodes method after using remove_node method on
        Graph class object

        <b>Steps</b>
        1. Check values returned by nodes and edges properties and num_nodes method after using remove_node method by
        non-directed Graph class object
        2. Check values returned by nodes and edges properties and num_nodes method after using remove_node method by
        directed Graph class object
        """
        # Scenario for non-directed graph
        non_directed_graph = self.non_directed_graph()
        # Removing node that connected to one node of non-directed graph
        non_directed_graph.remove_node(1)
        assert non_directed_graph.num_nodes() == 3
        self.check_graph_list_attributes(
            [
                {
                    "attribute": non_directed_graph.edges,
                    "expected_type": EdgeDataView,
                    "expected_value": [(2, 3, {"value": 3}), (3, 4, {"value": 4})],
                },
                {
                    "attribute": non_directed_graph.nodes,
                    "expected_type": NodeView,
                    "expected_value": [2, 3, 4],
                },
            ]
        )
        # Checking that NetworkXError exception raised when removing node non existing nodes from non-directed graph
        for node in [1, 5]:
            with pytest.raises(NetworkXError):
                non_directed_graph.remove_node(node)
        # Removing node that connected to two nodes of non-directed graph
        non_directed_graph.remove_node(3)
        assert non_directed_graph.num_nodes() == 2
        self.check_graph_list_attributes(
            [
                {
                    "attribute": non_directed_graph.edges,
                    "expected_type": EdgeDataView,
                    "expected_value": [],
                },
                {
                    "attribute": non_directed_graph.nodes,
                    "expected_type": NodeView,
                    "expected_value": [2, 4],
                },
            ]
        )
        # Scenario for directed graph
        directed_graph = self.directed_graph()
        # Removing node that connected to two nodes of non-directed graph and has multi-direction edge
        directed_graph.remove_node("A")
        assert directed_graph.num_nodes() == 3
        self.check_graph_list_attributes(
            [
                {
                    "attribute": directed_graph.edges,
                    "expected_type": OutMultiEdgeDataView,
                    "expected_value": [
                        ("B", "C", 0, {"value": 2}),
                        ("C", "D", 0, {"value": 3}),
                    ],
                },
                {
                    "attribute": directed_graph.nodes,
                    "expected_type": NodeView,
                    "expected_value": ["B", "C", "D"],
                },
            ]
        )
        # Checking that NetworkXError exception raised when removing node non existing nodes from directed graph
        for node in ["A", "E"]:
            with pytest.raises(NetworkXError):
                directed_graph.remove_node(node)
        # Removing node that connected to two nodes of directed graph, it causes removal of all edges
        directed_graph.remove_node("C")
        assert non_directed_graph.num_nodes() == 2
        self.check_graph_list_attributes(
            [
                {
                    "attribute": directed_graph.edges,
                    "expected_type": OutMultiEdgeDataView,
                    "expected_value": [],
                },
                {
                    "attribute": directed_graph.nodes,
                    "expected_type": NodeView,
                    "expected_value": ["B", "D"],
                },
            ]
        )

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_graph_num_nodes(self):
        """
        <b>Description:</b>
        Check num_nodes method of Graph class object

        <b>Input data:</b>
        Graph objects with specified "directed" and "edges" parameters

        <b>Expected results:</b>
        Test passes if num_nodes method returns expected value

        <b>Steps</b>
        1. Check values returned by num_nodes method for non-directed Graph class object
        2. Check values returned by num_nodes method for directed Graph class object
        """
        for empty_graph in [Graph(), Graph(directed=True)]:
            assert empty_graph.num_nodes() == 0
        # Scenario for non-directed graph
        non_directed_graph = self.non_directed_graph()
        assert non_directed_graph.num_nodes() == 4
        # Checking num_nodes method after adding one node to non-directed graph
        non_directed_graph.add_node(5)
        assert non_directed_graph.num_nodes() == 5
        # Checking num_nodes method after adding edge with one existing node to non-directed graph
        non_directed_graph.add_edge(1, 6)
        assert non_directed_graph.num_nodes() == 6
        # Checking num_nodes method after adding edge with non-existing nodes to non-directed graph
        non_directed_graph.add_edge(7, 8)
        assert non_directed_graph.num_nodes() == 8
        # Checking num_nodes method after removing node from non-directed graph
        non_directed_graph.remove_node(1)
        assert non_directed_graph.num_nodes() == 7
        # Checking num_nodes method after removing edge from non-directed graph
        non_directed_graph.remove_edges(2, 3)
        assert non_directed_graph.num_nodes() == 7
        # Scenario for directed graph
        directed_graph = self.directed_graph()
        assert directed_graph.num_nodes() == 4
        # Checking num_nodes method after adding one node to directed graph
        directed_graph.add_node("E")
        assert directed_graph.num_nodes() == 5
        # Checking num_nodes method after adding edge with one existing node to directed graph
        directed_graph.add_edge("A", "F")
        assert directed_graph.num_nodes() == 6
        # Checking num_nodes method after adding edge with non-existing nodes to directed graph
        directed_graph.add_edge("G", "H")
        assert directed_graph.num_nodes() == 8
        # Checking num_nodes method after removing node from directed graph
        directed_graph.remove_node("A")
        assert directed_graph.num_nodes() == 7
        # Checking num_nodes method after removing edge from non-directed graph
        directed_graph.remove_edges("B", "C")
        assert directed_graph.num_nodes() == 7

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_graph_eq(self):
        """
        <b>Description:</b>
        Check __eq__ method of Graph class object

        <b>Input data:</b>
        Graph objects with specified "directed" and "edges" parameters

        <b>Expected results:</b>
        Test passes if __eq__ method returns expected value

        <b>Steps</b>
        1. Check values returned by __eq__ method for non-directed Graph class object
        2. Check values returned by __eq__ method for directed Graph class object
        """
        non_directed_graph = Graph()
        directed_graph = Graph(directed=True)
        unequal_nodes_graph = Graph()
        unequal_edges_graph = Graph()
        equal_non_directed_graph = Graph()
        graph_edges_list = [
            {"node1": 1, "node2": 2, "edge_value": 1},
            {"node1": 2, "node2": 3, "edge_value": 2},
        ]
        for no_edges_graph in [
            non_directed_graph,
            directed_graph,
            equal_non_directed_graph,
            unequal_nodes_graph,
            unequal_edges_graph,
        ]:
            self.add_edges_to_graph(no_edges_graph, graph_edges_list)

        # Checking __eq__ method for equal non-directed graphs
        assert non_directed_graph == equal_non_directed_graph
        # Checking __eq__ method for non-directed graphs with unequal directed attributes
        assert non_directed_graph != directed_graph
        # Checking __eq__ method for non-directed graphs with unequal nodes
        unequal_nodes_graph.add_node(4)
        assert non_directed_graph != unequal_nodes_graph
        # Checking __eq__ method for non-directed graphs with unequal edges
        unequal_edges_graph.add_edge(node1=1, node2=3, edge_value=4)
        assert non_directed_graph != unequal_edges_graph
        # Checking __eq__ method by comparing non-directed Graph with object of other type
        assert non_directed_graph != str
        # Check for non-directed graph
        equal_directed_graph = Graph(directed=True)
        unequal_nodes_graph = Graph(directed=True)
        unequal_edges_graph = Graph(directed=True)
        for no_edges_graph in [
            equal_directed_graph,
            unequal_nodes_graph,
            unequal_edges_graph,
        ]:
            self.add_edges_to_graph(no_edges_graph, graph_edges_list)
        # Checking __eq__ method for equal directed graphs
        assert directed_graph == equal_directed_graph
        # Checking __eq__ method for directed graphs with unequal directed attributes
        assert directed_graph != non_directed_graph
        # Checking __eq__ method for directed graphs with unequal nodes
        unequal_nodes_graph.add_node(4)
        assert directed_graph != unequal_nodes_graph
        # Checking __eq__ method for directed graphs with unequal edges
        unequal_edges_graph.add_edge(2, 1, 4)
        assert directed_graph != unequal_edges_graph
        # Checking __eq__ method by comparing directed Graph with object of other type
        assert directed_graph != str

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_graph_descendants(self):
        """
        <b>Description:</b>
        Check descendants method of Graph class object

        <b>Input data:</b>
        Graph objects with specified "directed" and "edges" parameters

        <b>Expected results:</b>
        Test passes if descendants method returns expected value

        <b>Steps</b>
        1. Check values returned by descendants method for non-directed Graph class object
        2. Check values returned by descendants method for directed Graph class object
        """
        # Scenario for non-directed graph
        non_directed_graph = self.non_directed_graph()
        assert non_directed_graph.descendants(1) == [1, 2, 3]
        assert non_directed_graph.descendants(2) == [2, 2, 3]
        assert non_directed_graph.descendants(3) == [3, 2, 3]
        assert non_directed_graph.descendants(4) == [4, 3, 2]
        # Checking descendants after removing node from non-directed graph
        non_directed_graph.remove_node(2)
        assert non_directed_graph.descendants(1) == []
        assert non_directed_graph.descendants(2) == []
        assert non_directed_graph.descendants(3) == [3]
        assert non_directed_graph.descendants(4) == [4]
        # Scenario for directed graph
        directed_graph = self.directed_graph()
        assert directed_graph.descendants("A") == ["D", "C", "B", "A", "A"]
        assert directed_graph.descendants("B") == ["A", "D", "C", "B", "A"]
        assert directed_graph.descendants("C") == ["B", "A", "D", "C", "A"]
        assert directed_graph.descendants("D") == ["C", "B", "A", "D", "A"]
        # Checking descendants after removing node from directed graph
        directed_graph.remove_node("B")
        assert directed_graph.descendants("A") == ["D", "C", "A"]
        assert directed_graph.descendants("B") == []
        assert directed_graph.descendants("C") == []
        assert directed_graph.descendants("D") == ["C", "A", "D"]


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestMultiDiGraph:
    @staticmethod
    def multi_di_graph():
        multi_di_graph = MultiDiGraph()
        TestGraph.add_edges_to_graph(
            multi_di_graph,
            [
                {"node1": (1, 1), "node2": (1, 2)},
                {"node1": (1, 2), "node2": (3, 1)},
                {"node1": (3, 1), "node2": (2, 1)},
            ],
        )
        return multi_di_graph

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
    def test_multi_di_graph(self):
        """
        <b>Description:</b>
        Check MultiDiGraph class object initialization

        <b>Input data:</b>
        MultiDiGraph object with specified "edges" parameter

        <b>Expected results:</b>
        Test passes if MultiDiGraph object "directed" attribute, "edges", "nodes" and "num_labels" properties and
        "num_nodes" method return expected values
        """
        multi_di_graph = self.multi_di_graph()
        TestGraph.check_graph_non_list_attributes(
            [
                {"attribute": multi_di_graph.directed, "expected_value": True},
                {"attribute": multi_di_graph.num_labels, "expected_value": 4},
                {"attribute": multi_di_graph.num_nodes(), "expected_value": 4},
            ]
        )
        TestGraph.check_graph_list_attributes(
            [
                {
                    "attribute": multi_di_graph.nodes,
                    "expected_type": NodeView,
                    "expected_value": [(1, 1), (1, 2), (3, 1), (2, 1)],
                },
                {
                    "attribute": multi_di_graph.edges,
                    "expected_type": OutMultiEdgeDataView,
                    "expected_value": [
                        ((1, 1), (1, 2), 0, {"value": None}),
                        ((1, 2), (3, 1), 0, {"value": None}),
                        ((3, 1), (2, 1), 0, {"value": None}),
                    ],
                },
            ]
        )

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_multi_di_graph_topological_sort(self):
        """
        <b>Description:</b>
        Check topological_sort method of MultiDiGraph class object

        <b>Input data:</b>
        MultiDiGraph objects with specified "edges" parameter

        <b>Expected results:</b>
        Test passes if topological_sort method returns generator object with expected values
        """
        multi_di_graph = self.multi_di_graph()
        topological_sort = multi_di_graph.topological_sort()
        for expected_value in [(1, 1), (1, 2), (3, 1), (2, 1)]:
            assert next(topological_sort) == expected_value
