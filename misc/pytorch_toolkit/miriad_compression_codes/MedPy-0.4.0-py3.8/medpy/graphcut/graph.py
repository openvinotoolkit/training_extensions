# Copyright (C) 2013 Oskar Maier
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# author Oskar Maier
# version r0.1.2
# since 2012-02-06
# status Release

# build-in modules

# third-party modules

# own modules
from .maxflow import GraphDouble, GraphFloat

# code
class Graph(object):
    r"""
    Represents a graph suitable for further processing with the graphcut package.
    
    The graph contains nodes, edges (directed) between the nodes (n-edges), edges
    between two terminals (called source and sink) and the nodes (t-edges), and a
    weight for each edge. 
    
    Notes
    -----
    The node-ids used by the graph are assumed to start with 1 and be
    continuous. This is not actually checked, except when calling the
    inconsistent() method, so be careful.
    """
    
    # @var __INT_16_BIT The maximum value of signed int 16bit.
    __INT_16_BIT = 32767
    # @var __UINT_16_BIT: The maximum value of unsigned int 16bit.
    __UINT_16_BIT = 65535
    # @var MAX The maximum value a weight can take.
    MAX = __UINT_16_BIT
    
    def __init__(self):
        self.__nodes = 0
        self.__snodes = []
        self.__tnodes = []
        self.__nweights = {}
        self.__tweights = {}
        
    def set_nodes(self, nodes):
        r"""
        Set the number of graph nodes (starting from node-id = 1),
        excluding sink and source.
        
        Parameters
        ----------
        nodes : int
            Number of nodes
        """
        self.__nodes = int(nodes)
        
    def set_source_nodes(self, source_nodes):
        r"""
        Set the source nodes and compute their t-weights.
        
        Parameters
        ----------
        source_nodes : sequence of integers
            Declare the source nodes via their ids.
        
        Notes
        -----
        It does not get checked if one of the supplied source-nodes already has
        a weight assigned (e.g. by passing it to `set_sink_nodes`). This can
        occur when the foreground- and background-markers cover the same region. In this
        case the order of setting the terminal nodes can affect the graph and therefore
        the graph-cut result.
        """
        self.__snodes = list(source_nodes)
        
        # set the source-to-node weights (t-weights)
        for snode in self.__snodes:
            self.__tweights[snode] = (self.MAX, 0) # (weight-to-source, weight-to-sink)
            
    def set_sink_nodes(self, sink_nodes):
        r"""
        Set the sink nodes and compute their t-weights.
        
        Parameters
        ----------
        sink_nodes : sequence of integers
            Declare the sink nodes via their ids.
        
        Notes
        -----
        It does not get checked if one of the supplied sink-nodes already has
        a weight assigned (e.g. by passing it to set_source_nodes()). This can
        occur when the foreground- and background-markers cover the same region. In this
        case the order of setting the terminal nodes can affect the graph and therefore
        the graph-cut result.
        """
        self.__tnodes = list(sink_nodes)
        
        # set the source-to-node weights (t-weights)
        for tnode in self.__tnodes:
            self.__tweights[tnode] = (0, self.MAX) # (weight-to-source, weight-to-sink)
            
    def set_nweights(self, nweights):
        r"""
        Sets all n-weights.
        
        Parameters
        ----------
        nweights : dict
            A dictionary with (node-id, node-id) tuples as keys and (weight-a-to-b, weight-b-to-a) as values.
        """
        self.__nweights = nweights
            
    def add_tweights(self, tweights):
        r"""
        Adds t-weights to the current collection of t-weights, overwriting already
        existing ones.
        
        Parameters
        ----------
        tweights : dict
            A dictionary with node_ids as keys and (weight-to-source, weight-to-sink) tuples as values.
        
        Notes
        -----
        The weights for nodes directly connected to either the source or the sink
        are best set using `set_source_nodes` or `set_sink_nodes` to ensure
        consistency of their maximum values.
        """
        self.__tweights.update(tweights)    
        
    def get_node_count(self):
        r"""
        Get the number of nodes.
        
        Returns
        -------
        node_count : int
            The number of nodes (excluding sink and source).
        """
        return self.__nodes
        
    def get_nodes(self):
        r"""
        Get the nodes.
        
        Returns
        -------
        nodes : list
            All nodes as an ordered list.
        """
        return list(range(1, self.__nodes + 1))
    
    def get_source_nodes(self):
        r"""
        Get the source nodes.
        
        Returns
        -------
        source_nodes : list
            All nodes that are connected with the source as an unordered list (excluding sink and source).
        """
        return self.__snodes
    
    def get_sink_nodes(self):
        r"""
        Get the sink nodes.
        
        Returns
        -------
        sink_nodes : list
            All nodes that are connected with the sink as an unordered list (excluding sink and source).
        """
        return self.__tnodes
    
    def get_edges(self):
        r"""
        Get the edges.
        
        Returns
        -------
        edges : list
            All edges as ordered list of tuples (i.e. [(node_id1, node_id2), (..), ...].
        """
        return list(self.__nweights.keys())
        
    def get_nweights(self):
        r"""
        Get the nweights.
        
        Returns
        -------
        nweights : dict
            All n-weights (inter-node weights) as {edge-tuple: (weight, weight_reverersed)...} dict.
        """
        return self.__nweights
        
    def get_tweights(self):
        r"""
        Get the tweights.
        
        Returns
        -------
        tweights : dict
            All t-weights (terminal-node weights) as {node_id: (weight-source-node, weight-node-sink), ...} dict.
        
        Notes
        -----
        Returns only the t-weights that have been set so far. For nodes with unset t-weight, no entry is returned.
        """
        return self.__tweights
    
    def inconsistent(self):
        r"""
        Perform some consistency tests on the graph represented by this object
        
        Returns
        -------
        consistent : bool or list
            False if consistent, else a list of inconsistency messages.
        
        Notes
        -----
        This check is very time intensive and should not be executed on huge
        graphs, except for debugging purposes.
        """
        messages = []
        for node in list(self.__tweights.keys()):
            if not node <= self.__nodes: messages.append("Node {} in t-weights but not in nodes.".format(node))
        for node in self.__snodes:
            if not node <= self.__nodes: messages.append("Node {} in s-nodes but not in nodes.".format(node))
        for node in self.__tnodes:
            if not node <= self.__nodes: messages.append("Node {} in t-nodes but not in nodes.".format(node))
        for e in list(self.__nweights.keys()):
            if not e[0] <= self.__nodes: messages.append("Node {} in edge {} but not in nodes.".format(e[0], e))
            if not e[1] <= self.__nodes: messages.append("Node {} in edge {} but not in nodes.".format(e[1], e))
            if (e[1], e[0]) in iter(list(self.__nweights.keys())): messages.append("The reversed edges of {} is also in the n-weights.".format(e))
                
            
        if 0 == len(messages): return False
        else: return messages
        
class GCGraph:
    r"""
    A graph representation that works directly with the maxflow.GraphDouble graph as
    base. It is therefore less flexible as graph.Graph, but leads to lower memory
    requirements.
    
    The graph contains nodes, edges (directed) between the nodes (n-edges), edges
    between two terminals (called source and sink) and the nodes (t-edges), and a
    weight for each edge. 
    
    Notes
    -----
    The node-ids used by the graph are assumed to start with 0 and be
    continuous. This is not actually checked, so be careful.
    
    This wrapper tries to catch the most usual exception that can occur in the
    underlying C++ implementation and to convert them into catchable and meaningful
    error messages.
    """
    # @var __INT_16_BIT The maximum value of signed int 16bit.
    __INT_16_BIT = 32767
    # @var __UINT_16_BIT: The maximum value of unsigned int 16bit.
    __UINT_16_BIT = 65535
    
    MAX = __UINT_16_BIT
    """The maximum value a terminal weight can take."""
    
    def __init__(self, nodes, edges):
        r"""
        Initialize.
        
        Parameters
        ----------
        nodes : int
            The number of nodes in the graph.
        edges : int
            The number of edges in the graph.
        """
        self.__graph = GraphDouble(nodes, edges)
        self.__graph.add_node(nodes)
        self.__nodes = nodes
        self.__edges = edges
        
    def set_source_nodes(self, source_nodes):
        r"""
        Set multiple source nodes and compute their t-weights.
        
        Parameters
        ----------
        source_nodes : sequence of integers
            Declare the source nodes via their ids.
        
        Raises
        ------
        ValueError 
            If a passed node id does not refer to any node of the graph
            (i.e. it is either higher than the initially set number of
            nodes or lower than zero).
                          
        Notes
        -----
        It does not get checked if one of the supplied source-nodes already has
        a weight assigned (e.g. by passing it to `set_sink_nodes`). This can
        occur when the foreground- and background-markers cover the same region. In this
        case the order of setting the terminal nodes can affect the graph and therefore
        the graph-cut result.
        """
        if max(source_nodes) >= self.__nodes or min(source_nodes) < 0:
            raise ValueError('Invalid node id of {} or {}. Valid values are 0 to {}.'.format(max(source_nodes), min(source_nodes), self.__nodes - 1))
        # set the source-to-node weights (t-weights)
        for snode in source_nodes:
            self.__graph.add_tweights(int(snode), self.MAX, 0) # (weight-to-source, weight-to-sink)
            
    def set_sink_nodes(self, sink_nodes):
        r"""
        Set multiple sink nodes and compute their t-weights.
        
        Parameters
        ----------
        sink_nodes : sequence of integers
            Declare the sink nodes via their ids.
            
        Raises
        ------
        ValueError 
            If a passed node id does not refer to any node of the graph
            (i.e. it is either higher than the initially set number of
            nodes or lower than zero).            
        
        Notes
        -----
        It does not get checked if one of the supplied sink-nodes already has
        a weight assigned (e.g. by passing it to `set_source_nodes`). This can
        occur when the foreground- and background-markers cover the same region. In this
        case the order of setting the terminal nodes can affect the graph and therefore
        the graph-cut result.
        """
        if max(sink_nodes) >= self.__nodes or min(sink_nodes) < 0:
            raise ValueError('Invalid node id of {} or {}. Valid values are 0 to {}.'.format(max(sink_nodes), min(sink_nodes), self.__nodes - 1))
        # set the node-to-sink weights (t-weights)
        for snode in sink_nodes:
            self.__graph.add_tweights(int(snode), 0, self.MAX) # (weight-to-source, weight-to-sink)
            
    def set_nweight(self, node_from, node_to, weight_there, weight_back):
        r"""
        Set a single n-weight / edge-weight.
        
        Parameters
        ----------
        node_from : int
            Node-id from the first node of the edge.
        node_to : int
            Node-id from the second node of the edge.
        weight_there : float
            Weight from first to second node (>0). 
        weight_back : float
            Weight from second to first node (>0).
        
        Raises
        ------
        ValueError
            If a passed node id does not refer to any node of the graph
            (i.e. it is either higher than the initially set number of
            nodes or lower than zero).
        ValueError
            If the two node-ids of the edge are the same (graph cut does
            not allow self-edges).
        ValueError
            If one of the passed weights is <= 0.
            
        Notes
        -----
        The object does not check if the number of supplied edges in total exceeds
        the number passed to the init-method. If this is the case, the underlying
        C++ implementation will double the memory, which is very unefficient.
        
        The underlying C++ implementation allows zero weights, but these are highly
        undesirable for inter-node weights and therefore raise an error.
        """
        if node_from >= self.__nodes or node_from < 0:
            raise ValueError('Invalid node id (node_from) of {}. Valid values are 0 to {}.'.format(node_from, self.__nodes - 1))
        elif node_to >= self.__nodes or node_to < 0:
            raise ValueError('Invalid node id (node_to) of {}. Valid values are 0 to {}.'.format(node_to, self.__nodes - 1))
        elif node_from == node_to:
            raise ValueError('The node_from ({}) can not be equal to the node_to ({}) (self-connections are forbidden in graph cuts).'.format(node_from, node_to))
        elif weight_there <= 0 or weight_back <= 0:
            raise ValueError('Negative or zero weights are not allowed.')
        self.__graph.sum_edge(int(node_from), int(node_to), float(weight_there), float(weight_back))
            
    def set_nweights(self, nweights):
        r"""
        Set multiple n-weights / edge-weights.
        
        Parameters
        ----------
        nweights : dict
            A dictionary with (node-id, node-id) tuples as keys and (weight-a-to-b, weight-b-to-a) as values.
        
        Notes
        -----
        The object does not check if the number of supplied edges in total exceeds
        the number passed to the init-method. If this is the case, the underlying
        C++ implementation will double the memory, which is very inefficient.
              
        See `set_nweight` for raised errors.
        """
        for edge, weight in list(nweights.items()):
            self.set_nweight(edge[0], edge[1], weight[0], weight[1])
            
    def set_tweight(self, node, weight_source, weight_sink):
        r"""
        Set a single t-weight / terminal-weight.
        
        Parameters
        ----------
        node : int
            Node-id for which to set the terminal weights.
        weight_source : float
            Weight to source terminal.
        weight_sink : float
            Weight to sink terminal.
        
        Raises
        ------
        ValueError
            If a passed node id does not refer to any node of the graph
            (i.e. it is either higher than the initially set number of
            nodes or lower than zero).
            
        Notes
        -----        
        The object does not check if the number of supplied edges in total exceeds
        the number passed to the init-method. If this is the case, the underlying
        C++ implementation will double the memory, which is very inefficient.
              
        Terminal weights can be zero or negative.
        """
        if node >= self.__nodes or node < 0:
            raise ValueError('Invalid node id of {}. Valid values are 0 to {}.'.format(node, self.__nodes - 1))
        self.__graph.add_tweights(int(node), float(weight_source), float(weight_sink)) # (weight-to-source, weight-to-sink)
            
    def set_tweights(self, tweights):
        r"""
        Set multiple t-weights to the current collection of t-weights, overwriting
        already existing ones.
        
        Parameters
        ----------
        tweights : dict
            A dictionary with node_ids as keys and (weight-to-source, weight-to-sink) tuples as values.
        
        Raises
        ------
        ValueError
            If a passed node id does not refer to any node of the graph
            (i.e. it is either higher than the initially set number of
            nodes or lower than zero).        
        
        Notes
        -----
        Since this method overrides already existing t-weights, it is strongly
        recommended to run `set_source_nodes` and `set_sink_nodes` after the
        last call to this method.
        
        The weights for nodes directly connected to either the source or the sink
        are best set using `set_source_nodes` or `set_sink_nodes` to ensure
        consistency of their maximum values.
        """        
        for node, weight in list(tweights.items()):
            self.set_tweight(node, weight[0], weight[1]) # (weight-to-source, weight-to-sink)
            
    def set_tweights_all(self, tweights):
        r"""
        Set all t-weights at once.
        
        Parameters
        ----------
        tweights : iterable
            Containing a pair of numeric values for each of the graphs nodes.
        
        Notes
        -----
        Since this method overrides already existing t-weights, it is strongly
        recommended to run `set_source_nodes` and `set_sink_nodes` after the
        last call to this method.
        
        The weights for nodes directly connected to either the source or the sink
        are best set using `set_source_nodes` or `set_sink_nodes` to ensure
        consistency of their maximum values.
        """
        for node, (twsource, twsink) in enumerate(tweights):
            self.set_tweight(node, twsource, twsink) # source = FG, sink = BG
        
    def get_graph(self):
        r"""
        Get the C++ graph.
        
        Returns
        -------
        graph : maxflow.GraphDouble
            The underlying maxflow.GraphDouble C++ implementation of the graph.
        """
        return self.__graph
        
    def get_node_count(self):
        r"""
        Get the number of nodes.
        
        Returns
        -------
        node_count : int
            The number of nodes (excluding sink and source).
        """
        return self.__nodes
        
    def get_nodes(self):
        r"""
        Get the nodes.
        
        Returns
        -------
        nodes : list
            All nodes as an ordered list (starting from 0).
        """
        return list(range(0, self.__nodes))
    
    def get_edge_count(self):
        r"""
        Get the number of edges.
        
        Returns
        -------
        edge_count : int
            The number of edges.
        """
        return self.__edges
