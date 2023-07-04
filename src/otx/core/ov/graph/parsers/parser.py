"""Parser modules for otx.core.ov.graph.parsers."""
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from typing import List


def type_parser(graph, types) -> List[str]:
    """Type Parser from graph, types."""
    found = []
    for node in graph:
        if node.type in types:
            found.append(node.name)
    return found


def result_parser(graph) -> List[str]:
    """Result Parser from graph."""
    return type_parser(graph, ["Result"])


def parameter_parser(graph) -> List[str]:
    """Parameter Parser from graph."""
    return type_parser(graph, ["Parameter"])
