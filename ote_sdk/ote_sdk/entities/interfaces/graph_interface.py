"""This module implements the Graph interface"""

# INTEL CONFIDENTIAL
#
# Copyright (C) 2021 Intel Corporation
#
# This software and the related documents are Intel copyrighted materials, and
# your use of them is governed by the express license under which they were provided to
# you ("License"). Unless the License provides otherwise, you may not use, modify, copy,
# publish, distribute, disclose or transmit this software or the related documents
# without Intel's prior written permission.
#
# This software and the related documents are provided as is,
# with no express or implied warranties, other than those that are expressly stated
# in the License.


import abc


class IGraph(metaclass=abc.ABCMeta):
    """
    This interface describes how the interface of the Graph looks like.
    This interface is used to represent the TaskGraph inside project as well as the label tree inside LabelSchema
    """

    @abc.abstractmethod
    def add_node(self, node):
        """
        Add node to the graph
        """
        raise NotImplementedError

    @abc.abstractmethod
    def add_edge(self, node1, node2):
        """
        Add an edge between node1 and node2
        """
        raise NotImplementedError

    @abc.abstractmethod
    def has_edge_between(self, node1, node2):
        """
        Returns whether there is an edge between `node1` and `node2`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def neighbors(self, node):
        """
        Returns neighbors of `node`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def find_cliques(self):
        """
        Returns cliques in the graph
        """
        raise NotImplementedError

    @abc.abstractmethod
    def num_nodes(self):
        """
        Returns number of nodes
        """
        raise NotImplementedError

    @abc.abstractmethod
    def remove_edges(self, node1, node2):
        """Removes the edges between two nodes"""
        raise NotImplementedError

    @abc.abstractmethod
    def find_out_edges(self, node):
        """
        Returns the edges coming out of the node
        """
        raise NotImplementedError

    @abc.abstractmethod
    def find_in_edges(self, node):
        """
        Returns the edges coming in to the node
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def edges(self):
        """
        Returns the edges in the Graph
        :return:
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def nodes(self):
        """
        Returns the nodes in the graph
        :return:
        """
        raise NotImplementedError
