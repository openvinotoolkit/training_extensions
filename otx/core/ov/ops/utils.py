"""Utils function for otx.core.ov.ops."""
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT


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
