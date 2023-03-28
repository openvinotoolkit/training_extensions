"""Functions for onnx adapters."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from functools import partial
from typing import Callable

import onnx
from onnx import ModelProto, NodeProto  # pylint: disable = no-name-in-module


def remove_nodes(model: ModelProto, predicate: Callable) -> ModelProto:
    """Remove nodes from ONNX model.

    Args:
        model (onnx.ModelProto): Input onnx model.
        predicate (Callable): A function to predicate a node.

    Returns:
        onnx.ModelProto: Modified onnx model.
    """
    # ! this doesn't handle inputs/outputs
    while True:
        connect = None
        for i, node in enumerate(model.graph.node):
            if predicate(node):
                assert len(node.input) == 1
                assert len(node.output) == 1
                connect = (node.input[0], node.output[0])
                del model.graph.node[i]
                break
        if not connect:
            break
        src, dst = connect
        for node in model.graph.node:
            for i, _input in enumerate(node.input):
                if _input == dst:
                    node.input[i] = src
    return model


def is_op(node: NodeProto, op_name) -> bool:
    """Check if an op is identity."""
    return node.op_type == op_name


def remove_node(model: ModelProto, op_name: str) -> ModelProto:  # noqa: C901
    """Remove identity node from an ONNX model.

    Args:
        model (onnx.ModelProto): Input onnx model.
        op_name (str): Operation name.
    """
    graph = model.graph

    def simplify_inputs():
        connect = None
        for _input in graph.input:
            for i, node in enumerate(graph.node):
                if node.op_type == op_name and node.input[0] == _input.name:
                    connect = (node.input[0], node.output[0])
                    del graph.node[i]
                    break
            if connect:
                break
        if not connect:
            return False
        src, dst = connect
        for node in graph.node:
            for i, input_name in enumerate(node.input):
                if input_name == dst:
                    node.input[i] = src
        # the input just changed won't be an output
        return True

    def simplify_outputs():
        connect = None
        for output in graph.output:
            for i, node in enumerate(graph.node):
                if node.op_type == op_name and node.output[0] == output.name:
                    connect = (node.input[0], node.output[0])
                    del graph.node[i]
                    break
            if connect:
                break
        if not connect:
            return False
        src, dst = connect
        for node in graph.node:
            for i, output_name in enumerate(node.output):
                if output_name == src:
                    node.output[i] = dst
            # the output just renamed may be someone's input
            for i, input_name in enumerate(node.input):
                if input_name == src:
                    node.input[i] = dst
        return True

    while simplify_inputs():
        pass

    while simplify_outputs():
        pass

    new_op = partial(is_op, op_name=op_name)
    remove_nodes(model, new_op)


def remove_nodes_by_op_type(onnx_model, op_type):
    """Remove all nodes of a specified op type from the ONNX model."""
    # TODO: support more nodes
    remove_node(onnx_model, op_type)
    onnx.checker.check_model(onnx_model)
    return onnx_model


def prepare_onnx_for_openvino(in_path, out_path):
    """Modify the specified ONNX model to be compatible with OpenVINO by removing 'Mark' op nodes."""
    onnx_model = onnx.load(in_path)
    onnx_model = remove_nodes_by_op_type(onnx_model, "Mark")
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, out_path)
