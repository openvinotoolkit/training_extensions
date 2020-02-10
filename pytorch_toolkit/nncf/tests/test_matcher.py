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

import networkx as nx

from nncf.dynamic_graph.graph_matching import NodeExpression as N, search_all


def add_nodes(graph, types, nodes=None):
    if nodes is None:
        nodes = list(range(1, len(types) + 1))
    for node, type_ in zip(nodes, types):
        graph.add_node(node, type=type_)


def test_simple():
    g = nx.DiGraph()
    add_nodes(g, ['a', 'b', 'c', 'a'])
    g.add_edges_from([(1, 2), (2, 3), (3, 4)])

    ex = N('b') + N('c')

    matches = search_all(g, ex)
    assert matches == [[2, 3]]


def test_two_matched():
    g = nx.DiGraph()
    add_nodes(g, ['a', 'b', 'c', 'a', 'b', 'c'])
    g.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)])

    ex = N('b') + N('c')

    matches = search_all(g, ex)
    assert matches == [[2, 3], [5, 6]]


def test_graph_branching():
    g = nx.DiGraph()
    add_nodes(g, ['a', 'b', 'a', 'c'])
    g.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 4)])

    ex = N('a') + N('b')

    matches = search_all(g, ex)
    assert matches == [[1, 2]]


def test_graph_branching_other_order():
    g = nx.DiGraph()
    add_nodes(g, ['a', 'a', 'b', 'c'])

    g.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 4)])

    ex = N('a') + N('b')

    matches = search_all(g, ex)
    assert matches == [[1, 3]]


def test_alternating():
    g = nx.DiGraph()
    add_nodes(g, ['a', 'b'])

    g.add_edges_from([(1, 2)])

    ex = N('a') + (N('a') | N('b'))

    matches = search_all(g, ex)
    assert matches == [[1, 2]]


def test_alternating_longest():
    g = nx.DiGraph()
    #   b c
    # a     d
    #    b
    add_nodes(g, ['a', 'b', 'c', 'b', 'd'])
    g.add_edges_from([(1, 2), (2, 3), (3, 5), (1, 4), (4, 5)])

    ex = N('a') + (N('b') | N('b') + N('c'))
    ex2 = N('a') + (N('b') + N('c') | N('b'))

    matches = search_all(g, ex)
    matches2 = search_all(g, ex2)

    assert matches2 == matches == [[1, 2, 3]]


def test_branching_expression():
    g = nx.DiGraph()
    #   b
    # a   d
    #   c
    add_nodes(g, ['a', 'b', 'c', 'd'])
    g.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 4)])
    c_ = (N('b') & N('c'))
    n = N('a')
    node_expression = N('d')
    ex = n + c_ + node_expression

    ex = N('a') + (N('b') & N('c')) + N('d')

    matches = search_all(g, ex)
    assert matches == [[1, 2, 3, 4]]


def test_branching_expression3():
    g = nx.DiGraph()
    #   b
    # a   d
    #   c
    add_nodes(g, ['a', 'b', 'c', 'd'])
    g.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 4)])
    c_ = N('b') & (N('e') | N('c'))
    n = N('a')
    node_expression = N('d')
    ex = n + c_ + node_expression

    ex = N('a') + (N('b') & N('c')) + N('d')

    matches = search_all(g, ex)
    assert matches == [[1, 2, 3, 4]]


def test_branching_expression2():
    g = nx.DiGraph()
    #   b
    # a e  d
    #   c
    add_nodes(g, ['a', 'b', 'c', 'd', 'e'])
    g.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 4), (1, 5), (5, 4)])
    c_ = (N('b') & N('c') & N('e'))
    n = N('a')
    node_expression = N('d')
    ex = n + c_ + node_expression

    matches = search_all(g, ex)
    assert matches == [[1, 2, 3, 5, 4]]
