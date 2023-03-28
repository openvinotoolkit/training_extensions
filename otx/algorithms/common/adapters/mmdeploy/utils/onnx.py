"""Functions for onnx adapters."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import onnx


def remove_nodes_by_op_type(onnx_model, op_type):
    """Remove all nodes of a specified op type from the ONNX model."""
    # TODO: support more nodes

    supported_op_types = ["Mark", "Conv", "Gemm"]
    assert op_type in supported_op_types

    target_nodes = []
    for node in onnx_model.graph.node:
        if node.op_type == op_type:
            target_nodes.append(node)

    output_names = [i.name for i in onnx_model.graph.output]

    for target_node in target_nodes:
        in_port = target_node.input[0]
        out_port = target_node.output[0]

        out_nodes = [node for node in onnx_model.graph.node if out_port in node.input]
        for out_node in out_nodes:
            for i, j in enumerate(out_node.input):
                if j == out_port:
                    out_node.input[i] = in_port

        if out_port in output_names:
            in_nodes = [node for node in onnx_model.graph.node if in_port in node.output]
            for in_node in in_nodes:
                for i, j in enumerate(in_node.output):
                    if j == in_port:
                        in_node.output[i] = out_port

        onnx_model.graph.node.remove(target_node)

    onnx.checker.check_model(onnx_model)
    return onnx_model


def prepare_onnx_for_openvino(in_path, out_path):
    """Modify the specified ONNX model to be compatible with OpenVINO by removing 'Mark' op nodes."""
    onnx_model = onnx.load(in_path)
    onnx_model = remove_nodes_by_op_type(onnx_model, "Mark")
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, out_path)
