# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from otx.v2.adapters.openvino.graph.graph import Graph
from otx.v2.adapters.openvino.graph.utils import (
    get_constant_input_nodes,
    handle_paired_batchnorm,
    handle_reshape,
)
from otx.v2.adapters.openvino.utils import load_ov_model


def get_graph(name: str="mobilenet-v2-pytorch") -> Graph:
    model = load_ov_model(f"omz://{name}")
    return Graph.from_ov(model)



def test_get_constant_input_nodes() -> None:
    graph = get_graph()
    node = graph.get_nodes_by_types(["Add"])[0]
    nodes = get_constant_input_nodes(graph, node)
    assert all(node.type == "Constant" for node in nodes)



def test_handle_merging_into_batchnorm() -> None:
    # TODO:
    pass


def test_handle_paired_batchnorm() -> None:
    graph = get_graph()
    handle_paired_batchnorm(graph)
    n_nodes = len(graph)
    assert graph.get_nodes_by_types(["BatchNormInference"])

    graph = get_graph()
    handle_paired_batchnorm(graph, replace=True)
    assert graph.get_nodes_by_types(["BatchNormInference"])

    assert n_nodes > len(graph)



def test_handle_reshape() -> None:
    graph = get_graph("dla-34")
    handle_reshape(graph)
