"""Utils function for otx.core.ov.ops."""
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from openvino.pyopenvino import Node  # pylint: disable=no-name-in-module

from .builder import OPS

# TODO: We moved the location of otx.mpa.utils.logger, we need to revert the logger in that code again.


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
    op_version = op_node.get_version()

    try:
        torch_module = OPS.get_by_type_version(op_type, op_version).from_ov(op_node)
    except Exception as e:
        # logger.error(e)
        # logger.error(op_type)
        # logger.error(op_version)
        # logger.error(op_node.get_attributes())
        raise e

    return torch_module
