# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from copy import deepcopy
from random import shuffle

import numpy as np
import openvino.runtime as ov
import pytest

from otx.core.ov.graph.graph import Graph, SortedDict
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestSortedDict:
    @e2e_pytest_unit
    def test(self):
        instance = SortedDict("key")
        orders = list("abcdefghijklmnopqrstuvwxyz")
        cands = list("abcdefghijklmnopqrstuvwxyz")
        shuffle(cands)

        for cand in cands:
            instance[cand] = {"edge": {"key": ord(cand)}}

        idx = 0
        for key in instance:
            assert key == orders[idx]
            idx += 1

        idx = len(orders) - 1
        for key in reversed(instance):
            assert key == orders[idx]
            idx -= 1

        repr(instance.keys())
        idx = 0
        for key in instance.keys():
            assert key == orders[idx]
            idx += 1

        idx = len(orders) - 1
        for key in reversed(instance.keys()):
            assert key == orders[idx]
            idx -= 1

        repr(instance.values())
        idx = 0
        for value in instance.values():
            assert value["edge"]["key"] == ord(orders[idx])
            idx += 1

        idx = len(orders) - 1
        for value in reversed(instance.values()):
            assert value["edge"]["key"] == ord(orders[idx])
            idx -= 1

        repr(instance.values())
        idx = 0
        for key, value in instance.items():
            assert key == orders[idx]
            assert value["edge"]["key"] == ord(orders[idx])
            idx += 1

        idx = len(orders) - 1
        for key, value in reversed(instance.items()):
            assert key == orders[idx]
            assert value["edge"]["key"] == ord(orders[idx])
            idx -= 1

        instance2 = deepcopy(instance)
        idx = 0
        for key, value in instance2.items():
            assert key == orders[idx]
            assert value["edge"]["key"] == ord(orders[idx])
            idx += 1

        instance.pop("i")
        assert "i" not in instance
        assert len(instance) == len(orders) - 1

        instance.clear()
        assert len(instance) == 0


class TestGraph:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        param = ov.opset10.parameter([1, 3, 64, 64], ov.Type.f32, name="in")
        constant = ov.opset10.constant(np.array([103.0, 116.0, 123.0]).reshape(1, 3, 1, 1), ov.Type.f32)
        node = ov.opset10.subtract(param, constant, "numpy")
        constant = ov.opset10.constant(np.random.normal(size=(32, 3, 3, 3)), ov.Type.f32)
        node = ov.opset10.convolution(node, constant, [2, 2], [1, 1], [1, 1], [1, 1], "explicit")
        constant = ov.opset10.constant(np.random.normal(size=(1, 32, 1, 1)), ov.Type.f32)
        node = ov.opset10.add(node, constant, "numpy")
        node = ov.opset10.clamp(node, 0, 6)
        result = ov.opset10.result(node, name="out")
        ov_model = ov.Model([result], [param], "model")

        self.graph = Graph.from_ov(ov_model)
        assert isinstance(self.graph, Graph)

    @e2e_pytest_unit
    def test_get_edge_data(self):
        nodes = [node for node in self.graph]
        assert self.graph.get_edge_data(nodes[0], nodes[-1]) is None
        assert self.graph.get_edge_data(nodes[0], nodes[2])

    @e2e_pytest_unit
    def test_remove_node(self):
        node = self.graph.get_nodes_by_types(["Subtract"])[0]
        predecessor = list(self.graph.predecessors(node))[0]
        successor = list(self.graph.successors(node))[0]
        self.graph.remove_node(node, keep_connect=True)
        assert self.graph.get_edge_data(predecessor, successor)

        node = self.graph.get_nodes_by_types(["Convolution"])[0]
        predecessor = list(self.graph.predecessors(node))[0]
        successor = list(self.graph.successors(node))[0]
        self.graph.remove_node(node, keep_connect=False)
        assert self.graph.get_edge_data(predecessor, successor) is None

    @e2e_pytest_unit
    def test_replace_node(self):
        node = self.graph.get_nodes_by_types(["Subtract"])[0]
        new_node = deepcopy(node)
        predecessors = list(self.graph.predecessors(node))
        successors = list(self.graph.successors(node))
        self.graph.replace_node(node, new_node)

        assert node not in self.graph
        assert new_node in self.graph
        assert predecessors == list(self.graph.predecessors(new_node))
        assert successors == list(self.graph.successors(new_node))

    @e2e_pytest_unit
    def test_add_edge(self):
        node = self.graph.get_nodes_by_types(["Subtract"])[0]
        new_node = deepcopy(node)
        predecessors = list(self.graph.predecessors(node))
        successors = list(self.graph.successors(node))
        self.graph.remove_node(node)

        for predecessor in predecessors:
            assert self.graph.get_edge_data(predecessor, new_node) is None
            self.graph.add_edge(predecessor, new_node)
            assert self.graph.get_edge_data(predecessor, new_node)

        for successor in successors:
            assert self.graph.get_edge_data(new_node, successor) is None
            self.graph.add_edge(new_node, successor)
            assert self.graph.get_edge_data(new_node, successor)

        assert new_node in self.graph

    @e2e_pytest_unit
    def test_get_nodes_by_type_pattern(self):
        node = self.graph.get_nodes_by_types(["Subtract"])[0]
        founds = self.graph.get_nodes_by_type_pattern(["Subtract", "Clamp"], node)
        for found in founds:
            start, end = found
            assert start == node
            assert start.type == "Subtract"
            assert end.type == "Clamp"

    @e2e_pytest_unit
    def test_remove_normalize_nodes(self):
        self.graph.remove_normalize_nodes()
        assert len(self.graph._normalize_nodes) == 0

    @e2e_pytest_unit
    def test_topological_sort(self):
        assert len(list(self.graph.topological_sort())) == len(self.graph)

    @e2e_pytest_unit
    def test_clean_up(self):
        nodes = self.graph.get_nodes_by_types(["Subtract"])
        self.graph.remove_node(nodes[0])
        n_nodes = len(self.graph)
        self.graph.clean_up()

        assert n_nodes > len(self.graph)
