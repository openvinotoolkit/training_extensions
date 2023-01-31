# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import List


def type_parser(graph, types) -> List[str]:
    found = []
    for node in graph:
        if node.type in types:
            found.append(node.name)
    return found


def result_parser(graph) -> List[str]:
    return type_parser(graph, ["Result"])


def parameter_parser(graph) -> List[str]:
    return type_parser(graph, ["Parameter"])
