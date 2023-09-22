"""Utils function for otx.v2.adapters.openvino.ops."""
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Union

import torch
from openvino.runtime import Node  # pylint: disable=no-name-in-module

# TODO: We moved the location of otx.mpa.utils.logger, we need to revert the logger in that code again.


def get_dynamic_shape(output: Node) -> list:
    """Getter function for dynamic shape."""
    shape: List[Union[str, int]] = [str(i) for i in output.get_partial_shape()]
    for i, shape_ in enumerate(shape):
        try:
            _shape = int(shape_)
        except ValueError:
            _shape = -1
        shape[i] = _shape
    return shape


def convert_op_to_torch(op_node: Node) -> torch.nn.Module:
    """Convert op Node to torch."""
    op_type = op_node.get_type_name()
    op_version = op_node.get_version()

    try:
        from .builder import OPS

        torch_module = OPS.get_by_type_version(op_type, op_version).from_ov(op_node)
    except Exception as e:
        raise e

    return torch_module
