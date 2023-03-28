# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from otx.core.ov.graph import Graph
from otx.core.ov.graph.parsers.parser import parameter_parser, result_parser
from otx.core.ov.utils import load_ov_model
from tests.test_suite.e2e_test_system import e2e_pytest_unit


def get_graph(name="mobilenet-v2-pytorch"):
    model = load_ov_model(f"omz://{name}")
    return Graph.from_ov(model)


@e2e_pytest_unit
def test_result_parser():
    graph = get_graph()
    names = result_parser(graph)
    names_ = [i.name for i in graph.get_nodes_by_types(["Result"])]
    assert set(names) == set(names_)


@e2e_pytest_unit
def test_parameter_parser():
    graph = get_graph()
    names = parameter_parser(graph)
    names_ = [i.name for i in graph.get_nodes_by_types(["Parameter"])]
    assert set(names) == set(names_)
