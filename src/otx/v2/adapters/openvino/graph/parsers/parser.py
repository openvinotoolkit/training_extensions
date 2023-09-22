"""Parser modules for otx.v2.adapters.openvino.graph.parsers."""
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

from otx.v2.adapters.openvino.graph import Graph


def type_parser(graph: Graph, types: list) -> List[str]:
    """Type Parser from graph, types."""
    found = []
    for node in graph:
        if node.type in types:
            found.append(node.name)
    return found


def result_parser(graph: Graph) -> List[str]:
    """Result Parser from graph."""
    return type_parser(graph, ["Result"])


def parameter_parser(graph: Graph) -> List[str]:
    """Parameter Parser from graph."""
    return type_parser(graph, ["Parameter"])
