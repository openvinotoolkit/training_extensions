# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from otx.core.ov.graph import Graph
from otx.core.ov.graph.parsers.cls import cls_base_parser
from otx.core.ov.utils import load_ov_model
from tests.test_suite.e2e_test_system import e2e_pytest_unit


def get_graph(name="mobilenet-v2-pytorch"):
    model = load_ov_model(f"omz://{name}")
    return Graph.from_ov(model)


@e2e_pytest_unit
def test_cls_base_parser():
    graph = get_graph()
    assert {
        "inputs": ["data"],
        "outputs": ["Clip_166"],
    } == cls_base_parser(graph, "backbone")
    assert {
        "inputs": ["GlobalAveragePool_167"],
        "outputs": ["Reshape_173"],
    } == cls_base_parser(graph, "neck")
    assert {
        "inputs": ["Gemm_174/WithoutBiases"],
        "outputs": ["prob/sink_port_0"],
    } == cls_base_parser(graph, "head")
