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

from itertools import chain, combinations
from typing import Callable, List

import numpy as np
import networkx as nx


def powerset(iterable, min_r=1, max_r=None):
    if not isinstance(iterable, list):
        s = list(iterable)
    else:
        s = iterable
    if max_r is None:
        max_r = len(s)
    return chain.from_iterable(combinations(s, r) for r in range(min_r, max_r + 1))


class Expression:
    def _match(self, nodes, graph):
        return NotImplementedError

    def __add__(self, other):
        return ConcatExpression([self, other])

    def __or__(self, other):
        return AlternatingExpression([self, other])

    def __and__(self, other):
        return BranchingExpression([self, other])

    def _iterate_alternatives(self, nodes):
        return powerset(nodes, min_r=1)

    def match(self, nodes, graph):
        all_matches = []
        for n in self._iterate_alternatives(nodes):
            result = self._match(n, graph)
            if not result:
                continue

            n, following = result
            following = list(following)
            if not isinstance(n, list):
                n = [n]

            all_matches.append((n, following))
        if not all_matches:
            return None, None
        return max(all_matches, key=lambda x: len(x[0]))


class ConcatExpression(Expression):
    def __init__(self, expressions):
        self.expressions = expressions

    def _match(self, nodes, graph):
        assert len(self.expressions) > 1
        full_match = []
        for ex in self.expressions:
            matches, following = ex.match(nodes, graph)

            if not matches:
                return None

            full_match += matches

            nodes = following
        return full_match, following

    def __add__(self, other):
        return ConcatExpression(self.expressions + [other])


class AlternatingExpression(Expression):
    def __init__(self, expressions, greedy_match=False, greedy_consume=True):
        self.greedy_match = greedy_match
        self.greedy_consume = greedy_consume
        self.expressions = expressions

    def _match(self, nodes, graph):
        assert len(self.expressions) > 1
        all_matches = []
        for ex in self.expressions:
            matched, following = ex.match(nodes, graph)
            if not matched:
                continue

            if self.greedy_match:
                return matched, following

            all_matches.append((matched, following))

        if self.greedy_consume:
            if not all_matches:
                return None
            return max(all_matches, key=lambda x: len(x[0]))
        return None

    def __or__(self, other):
        return AlternatingExpression(self.expressions + [other])


class BranchingExpression(Expression):
    def __init__(self, expressions):
        self.expressions = expressions

    def _iterate_alternatives(self, nodes):
        return powerset(nodes, len(self.expressions), len(self.expressions))

    def _match(self, nodes, graph):
        assert len(self.expressions) > 1
        if len(nodes) != len(self.expressions):
            # need to try all possible assignments
            return None

        matches = [[] for _ in range(len(self.expressions))]
        for i, ex in enumerate(self.expressions):
            any_matched = False
            for node_name in nodes:
                matched, following = ex.match([node_name], graph)
                matches[i].append((matched, following))

                if matched:
                    any_matched = True

            if not any_matched:
                return None

        return self._assign_matches(matches)

    def _assign_matches(self, matches):
        """Assign every expression to some match"""
        assignments = np.full(len(matches[0]), -1)
        used = np.full(len(matches), False)

        def _find_assignment(i):
            if used[i]:
                return False
            used[i] = True
            for j in range(len(matches[0])):
                if not matches[i][j][0]:
                    continue
                if assignments[j] == -1 or _find_assignment(assignments[j]):
                    assignments[j] = i
                    return True
            return False

        for i in range(len(self.expressions)):
            used[...] = False
            _find_assignment(i)

        all_matches = set()
        all_followings = set()
        for i in range(len(matches[0])):
            if assignments[i] != -1:
                match, follow = matches[assignments[i]][i]
                all_matches.update(match)
                all_followings.update(follow)

        # assume matches dot not end in other match
        if all_matches & all_followings:
            return None
        return list(all_matches), list(all_followings)

    def __and__(self, other):
        return BranchingExpression(self.expressions + [other])


class NodeExpression(Expression):
    def __init__(self, node_type: str = None, filter_fn=None, node_type_fn: Callable[[dict], str] = None):
        self.filter = filter_fn
        self.node_type = node_type
        if node_type_fn is None:
            self.node_type_fn = lambda x: x['type']
        else:
            self.node_type_fn = node_type_fn

    def _iterate_alternatives(self, nodes):
        for node in nodes:
            yield [node]

    def _match(self, nodes, graph):
        if len(nodes) != 1:
            return None

        node_name = nodes[0]
        node = graph.nodes[node_name]
        node_type = self.node_type_fn(node)
        if self.node_type == node_type:
            if self.filter and not self.filter(node):
                return None

            following = graph.successors(node_name)
            return node_name, following
        return None


def get_edge_boundaries(match: List[str], graph: nx.DiGraph):
    out_edge_boundary = list(nx.edge_boundary(graph, match, data=True))
    complement = list(filter(lambda x: x not in match, graph.nodes.keys()))
    in_edge_boundary = list(nx.edge_boundary(graph, complement, data=True))
    return in_edge_boundary, out_edge_boundary


def search_all(graph: nx.DiGraph, expression: Expression) -> List[List[str]]:
    """Returns list of node key lists that match the expression."""
    matches = []
    matched_nodes = set()
    weakly_subgraphs = [graph.subgraph(c) for c in nx.weakly_connected_components(graph)]
    for subgraph in weakly_subgraphs:
        dfs_order = nx.topological_sort(subgraph)
        for node in dfs_order:
            match, _ = expression.match([node], graph)

            if node in matched_nodes:
                continue

            if match:
                for mn in match:
                    matched_nodes.add(mn)
                matches.append(match)
    return matches
