"""Utils function for otx.core.ov.ops."""
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from openvino.runtime import Node

from .builder import OPS


def get_dynamic_shape(output):
    """Getter function for dynamic shape."""
    shape = [str(i) for i in output.get_partial_shape()]
    for i, shape_ in enumerate(shape):
        try:
            shape_ = int(shape_)
        except ValueError:
            shape_ = -1
        shape[i] = shape_
    return shape


def convert_op_to_torch(op_node: Node):
    """Convert op Node to torch."""
    op_type = op_node.get_type_name()

    op_version = op_node.get_type_info().version_id
    try:
        torch_module = OPS.get_by_type_version(op_type, op_version).from_ov(op_node)
    except Exception as e:
        raise e

    return torch_module
