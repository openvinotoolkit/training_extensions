"""Class base parser for otx.core.ov.graph.parsers.cls.cls_base_parser."""
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Optional

from otx.utils.logger import get_logger

from ..builder import PARSERS
from ..parser import parameter_parser

# pylint: disable=too-many-return-statements, too-many-branches

logger = get_logger()

NECK_INPUT_TYPES = ["ReduceMean", "MaxPool", "AvgPool"]
NECK_TYPES = [
    "Reshape",
    "Squeeze",
    "Unsqueeze",
    "Concat",
    "Convert",
    "ShapeOf",
    "StridedSlice",
    "Transpose",
]


@PARSERS.register()
def cls_base_parser(graph, component: str = "backbone") -> Optional[Dict[str, List[str]]]:
    """Class base parser for OMZ models."""
    assert component in ["backbone", "neck", "head"]

    result_nodes = graph.get_nodes_by_types(["Result"])
    if len(result_nodes) != 1:
        logger.debug("More than one reulst nodes are found.")
        return None
    result_node = result_nodes[0]

    neck_input = None
    for _, node_to in graph.bfs(result_node, True, 20):
        if node_to.type in NECK_INPUT_TYPES:
            logger.debug(f"Found neck_input: {node_to.name}")
            neck_input = node_to
            break

    if neck_input is None:
        # logger.debug("Can not determine the output of backbone.")
        return None

    neck_output = neck_input
    for node_from, node_to in graph.bfs(neck_input, False, 10):
        done = False
        for node_to_ in node_to:
            if node_to_.type not in NECK_TYPES:
                done = True
                break
        neck_output = node_from
        if done:
            break

    if component == "backbone":
        outputs = [node.name for node in graph.predecessors(neck_input) if node.type != "Constant"]
        if len(outputs) != 1:
            logger.debug(f"neck_input {neck_input.name} has more than one predecessors.")
            return None

        inputs = parameter_parser(graph)
        if len(inputs) != 1:
            logger.debug("More than on parameter nodes are found.")
            return None

        return dict(
            inputs=inputs,
            outputs=outputs,
        )

    if component == "neck":
        return dict(
            inputs=[neck_input.name],
            outputs=[neck_output.name],
        )

    if component == "head":
        head_inputs = list(graph.successors(neck_output))

        outputs = graph.get_nodes_by_types(["Result"])
        if len(outputs) != 1:
            logger.debug("More than one network output is found.")
            return None
        for node_from, node_to in graph.bfs(outputs[0], True, 5):
            if node_to.type == "Softmax":
                outputs = [node_from]
                break

        if not graph.has_path(head_inputs[0], outputs[0]):
            logger.debug(f"input({head_inputs[0].name}) and output({outputs[0].name}) are reversed")
            return None

        return dict(
            inputs=[input_.name for input_ in head_inputs],
            outputs=[output.name for output in outputs],
        )
    return None
