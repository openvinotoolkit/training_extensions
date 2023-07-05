"""This module implements the TrainParameters entity."""
# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import abc
from typing import Iterator, List, Union

import networkx as nx


class IGraph(metaclass=abc.ABCMeta):
    """This interface describes how the interface of the Graph looks like.

    This interface is used to represent the TaskGraph inside project as well as the label tree inside LabelSchema
    """

    @abc.abstractmethod
    def add_node(self, node):
        """Add node to the graph."""
        raise NotImplementedError

    @abc.abstractmethod
    def add_edge(self, node1, node2):
        """Add an edge between node1 and node2."""
        raise NotImplementedError

    @abc.abstractmethod
    def has_edge_between(self, node1, node2):
        """Returns whether there is an edge between `node1` and `node2`."""
        raise NotImplementedError

    @abc.abstractmethod
    def neighbors(self, node) -> List[dict]:
        """Returns neighbors of `node`."""
        raise NotImplementedError

    @abc.abstractmethod
    def find_cliques(self) -> Iterator[List[dict]]:
        """Returns cliques in the graph."""
        raise NotImplementedError

    @abc.abstractmethod
    def num_nodes(self) -> int:
        """Returns number of nodes."""
        raise NotImplementedError

    @abc.abstractmethod
    def remove_edges(self, node1, node2) -> None:
        """Removes the edges between two nodes."""
        raise NotImplementedError

    @abc.abstractmethod
    def find_out_edges(self, node) -> nx.reportviews.OutMultiEdgeView:
        """Returns the edges coming out of the node."""
        raise NotImplementedError

    @abc.abstractmethod
    def find_in_edges(self, node) -> nx.reportviews.InMultiEdgeView:
        """Returns the edges coming in to the node."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def edges(self) -> nx.reportviews.OutMultiEdgeView:
        """Returns the edges in the Graph."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def nodes(self) -> nx.reportviews.NodeView:
        """Return nodes in the graph."""
        raise NotImplementedError


class Graph(IGraph):
    """The concrete implementation of IGraph. This implementation is using networkx library.

    Args:
        directed (bool): set to True if the graph is a directed graph.
    """

    def __init__(self, directed: bool = False):
        self._graph: Union[nx.Graph, nx.MultiDiGraph] = nx.Graph() if not directed else nx.MultiDiGraph()
        self.directed = directed

    def get_graph(self) -> Union[nx.Graph, nx.MultiDiGraph]:
        """Get the underlying NetworkX graph."""
        return self._graph

    def set_graph(self, graph: Union[nx.Graph, nx.MultiDiGraph]):
        """Set the underlying NetworkX graph."""
        self._graph = graph

    def add_edge(self, node1, node2, edge_value=None):
        """Adds edge between node1 and node2."""
        # pylint: disable=arguments-differ
        self._graph.add_edge(node1, node2, value=edge_value)

    def num_nodes(self) -> int:
        """Returns the number of nodes in the graph."""
        return self._graph.number_of_nodes()

    def add_node(self, node):
        """Adds node to the graph."""
        if node not in self._graph.nodes:
            self._graph.add_node(node)

    def has_edge_between(self, node1, node2):
        """Returns True if there is an edge between node1 and node2."""
        return node1 in self.neighbors(node2)

    def neighbors(self, node):
        """Returns neighbors of `label`.

        Note: when `node` does not exist in the graph an empty list is returned
        """
        try:
            result = list(self._graph.neighbors(node))
        except nx.NetworkXError:
            result = []
        return result

    def find_out_edges(self, node):
        """Returns the edges that have `node` as a destination."""
        # pylint: disable=no-member
        if node not in self._graph.nodes:
            raise KeyError(f"The node `{node}` is not part of the graph")

        if isinstance(self._graph, nx.MultiDiGraph):
            return self._graph.out_edges(node)
        return []

    def find_in_edges(self, node):
        """Returns the edges that have `node` as a source."""
        # pylint: disable=no-member
        if node not in self._graph.nodes:
            raise KeyError(f"The node `{node}` is not part of the graph")

        if isinstance(self._graph, nx.MultiDiGraph):
            return self._graph.in_edges(node)
        return []

    def find_cliques(self):
        """Returns cliques in the graph."""
        return nx.algorithms.clique.find_cliques(self._graph)

    @property
    def nodes(self):
        """Returns the nodes in the graph."""
        return self._graph.nodes

    @property
    def edges(self):
        """Returns all the edges in the graph."""
        if isinstance(self._graph, nx.MultiDiGraph):
            all_edges = self._graph.edges(keys=True, data=True)
        else:
            all_edges = self._graph.edges(data=True)
        return all_edges

    @property
    def num_labels(self):
        """Returns the number of labels in the graph."""
        return nx.convert_matrix.to_numpy_matrix(self._graph).shape[0]

    def remove_edges(self, node1, node2):
        """Removes edges between both the nodes."""
        self._graph.remove_edge(node1, node2)

    def remove_node(self, node):
        """Remove node from graph.

        Args:
            node: node to remove
        """
        self._graph.remove_node(node)

    def descendants(self, parent):
        """Returns descendants.

        (children and children of children, etc.) of `parent`.
        """
        try:
            edges = list(nx.edge_dfs(self._graph, parent, orientation="reverse"))
        except nx.exception.NetworkXError:
            edges = []
        return [edge[0] for edge in edges]

    def __eq__(self, other: object) -> bool:
        """Returns True if the two graphs are equal."""
        if isinstance(other, Graph):
            return (
                self.directed == other.directed
                and self._graph.nodes == other._graph.nodes
                and self._graph.edges == other._graph.edges
            )
        return False


class MultiDiGraph(Graph):
    """Multi Dimensional implementation of a Graph."""

    def __init__(self) -> None:
        super().__init__(directed=True)

    def topological_sort(self):
        """Returns a generator of nodes in topologically sorted order."""
        return nx.topological_sort(self._graph)
