# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import math
from functools import partial

from torch.nn import functional as F


def get_dynamic_shape(op):
    shape = [str(i) for i in op.get_partial_shape()]
    for i, shape_ in enumerate(shape):
        try:
            shape_ = int(shape_)
        except ValueError:
            shape_ = -1
        shape[i] = shape_
    return shape


def get_torch_padding(pads_begin, pads_end, auto_pad, input_size, weight_size, stride, dilation=None):
    from .movements import PadV1

    if dilation is None:
        dilation = [1 for _ in input_size]

    if auto_pad == "valid":
        return 0
    elif auto_pad == "same_upper" or auto_pad == "same_lower":
        assert len(set(dilation)) == 1 and dilation[0] == 1
        pads_begin = []
        pads_end = []
        for input_size_, weight_size_, stride_, dilation_ in zip(input_size, weight_size, stride, dilation):
            out_size = math.ceil(input_size_ / stride_)
            padding_needed = max(0, (out_size - 1) * stride_ + weight_size_ - input_size_)
            padding_lhs = int(padding_needed / 2)
            padding_rhs = padding_needed - padding_lhs

            pads_begin.append(padding_lhs if auto_pad == "same_upper" else padding_rhs)
            pads_end.append(padding_rhs if auto_pad == "same_upper" else padding_lhs)
        pad = PadV1.get_torch_pad_dim(pads_begin, pads_end)
        return partial(F.pad, pad=pad, mode="constant", value=0)
    elif auto_pad == "explicit":
        pad = PadV1.get_torch_pad_dim(pads_begin, pads_end)
        return partial(F.pad, pad=pad, mode="constant", value=0)
    else:
        raise NotImplementedError
