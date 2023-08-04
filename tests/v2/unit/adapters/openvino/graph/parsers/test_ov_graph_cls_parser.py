# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from otx.v2.adapters.openvino.graph import Graph
from otx.v2.adapters.openvino.graph.parsers.cls import cls_base_parser
from otx.v2.adapters.openvino.utils import load_ov_model


def get_graph(name: str="mobilenet-v2-pytorch") -> None:
    model = load_ov_model(f"omz://{name}")
    return Graph.from_ov(model)



def test_cls_base_parser() -> None:
    graph = get_graph()
    assert {
        "inputs": ["data"],
        "outputs": ["/features/features#18/features#18#2/Clip"],
    } == cls_base_parser(graph, "backbone")
    assert {
        "inputs": ["/GlobalAveragePool"],
        "outputs": ["/Flatten"],
    } == cls_base_parser(graph, "neck")
    assert {
        "inputs": ["/classifier/classifier#1/Gemm/WithoutBiases"],
        "outputs": ["prob/sink_port_0"],
    } == cls_base_parser(graph, "head")
