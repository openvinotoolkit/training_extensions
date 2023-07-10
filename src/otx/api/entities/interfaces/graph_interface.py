"""This module implements the Graph interface."""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


import abc
from typing import Iterator, List

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
