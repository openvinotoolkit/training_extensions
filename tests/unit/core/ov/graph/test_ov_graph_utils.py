# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest
from otx.core.ov.graph.graph import Graph
from otx.core.ov.graph.utils import (
    get_constant_input_nodes,
    handle_paired_batchnorm,
    handle_reshape,
)
from otx.core.ov.utils import load_ov_model
from tests.test_suite.e2e_test_system import e2e_pytest_unit


def get_graph(name="mobilenet-v2-pytorch"):
    model = load_ov_model(f"omz://{name}")
    return Graph.from_ov(model)


@e2e_pytest_unit
def test_get_constant_input_nodes():
    graph = get_graph()
    node = graph.get_nodes_by_types(["Add"])[0]
    nodes = get_constant_input_nodes(graph, node)
    assert all([node.type == "Constant" for node in nodes])


@e2e_pytest_unit
def test_handle_merging_into_batchnorm():
    # TODO:
    pass
    #  graph = get_graph()
    #  n_nodes = len(graph)
    #  handle_merging_into_batchnorm(graph)
    #
    #  assert graph.get_nodes_by_types(["BatchNormInference"])
    #  assert n_nodes >= len(graph)


@e2e_pytest_unit
@pytest.mark.skip(reason="Updated models are not compatible with the paired batchnorm converter")
def test_handle_paired_batchnorm():
    graph = get_graph()
    handle_paired_batchnorm(graph)
    n_nodes = len(graph)
    assert graph.get_nodes_by_types(["BatchNormInference"])

    graph = get_graph()
    handle_paired_batchnorm(graph, replace=True)
    assert graph.get_nodes_by_types(["BatchNormInference"])

    assert n_nodes > len(graph)


@e2e_pytest_unit
def test_handle_reshape():
    graph = get_graph("dla-34")
    handle_reshape(graph)
