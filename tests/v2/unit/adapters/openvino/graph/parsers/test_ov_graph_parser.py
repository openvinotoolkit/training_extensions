# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from otx.v2.adapters.openvino.graph import Graph
from otx.v2.adapters.openvino.graph.parsers.parser import parameter_parser, result_parser
from otx.v2.adapters.openvino.utils import load_ov_model


def get_graph(name: str="mobilenet-v2-pytorch") -> None:
    model = load_ov_model(f"omz://{name}")
    return Graph.from_ov(model)



def test_result_parser() -> None:
    graph = get_graph()
    names = result_parser(graph)
    names_ = [i.name for i in graph.get_nodes_by_types(["Result"])]
    assert set(names) == set(names_)



def test_parameter_parser() -> None:
    graph = get_graph()
    names = parameter_parser(graph)
    names_ = [i.name for i in graph.get_nodes_by_types(["Parameter"])]
    assert set(names) == set(names_)
