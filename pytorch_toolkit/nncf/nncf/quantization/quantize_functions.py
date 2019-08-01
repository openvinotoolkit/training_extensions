"""
 Copyright (c) 2019 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import warnings

import torch

from ..dynamic_graph import register_operator
from ..functions import STRound, clamp
from ..utils import is_tracing_state, no_jit_trace
from .extensions import QuantizedFunctionsCPU, QuantizedFunctionsCUDA


class QuantizeSymmetric(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, scale, level_low, level_high):
        if input_.is_cuda:
            output = QuantizedFunctionsCUDA.QuantizeSymmetric_forward(input_, scale, level_low, level_high)
        else:
            output = QuantizedFunctionsCPU.QuantizeSymmetric_forward(input_, scale, level_low, level_high)

        ctx.save_for_backward(input_, scale)
        ctx.level_low = level_low
        ctx.level_high = level_high

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, scale = ctx.saved_tensors
        level_low = ctx.level_low
        level_high = ctx.level_high

        if grad_output.is_cuda:
            if not grad_output.is_contiguous():
                warnings.warn("grad_output is not contiguous!", RuntimeWarning)
                grad_output = grad_output.contiguous()

            grad_input, grad_scale = QuantizedFunctionsCUDA.QuantizeSymmetric_backward(
                grad_output, input_, scale, level_low, level_high
            )
        else:
            grad_input, grad_scale = QuantizedFunctionsCPU.QuantizeSymmetric_backward(
                grad_output, input_, scale, level_low, level_high
            )

        return grad_input, grad_scale, None, None


def _quantize_autograd_to_range(x, input_low, input_high, levels):
    x = x - input_low
    input_range = (input_high - input_low)
    scale = (levels - 1) / input_range
    output = clamp(x, low=x.new_zeros(x.shape), high=input_range)
    output = output * scale
    output = STRound.apply(output)
    output = output * input_range / (levels - 1) + input_low
    return output


class ExportQuantize(torch.autograd.Function):
    @staticmethod
    def symbolic(g, x, levels, input_low, input_high, output_low, output_high):
        return g.op("FakeQuantize", x, input_low, input_high, output_low, output_high, levels_i=levels)

    @staticmethod
    def forward(ctx, x, levels, input_low, input_high, output_low, output_high):
        output = _quantize_autograd_to_range(x, input_low, input_high, levels)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # backward is not used during export
        return grad_output


@register_operator()
def quantize(x, scale, num_bits, bias=None, signed=False, is_weights=False):
    if signed:
        level_high = 2 ** (num_bits - 1) - 1
        level_low = -(level_high + 1)
        if is_weights:
            level_low += 1
    else:
        level_high = 2 ** num_bits - 1
        level_low = 0
    # todo: take bias into account during input_low/input_high calculation
    if is_tracing_state():
        with no_jit_trace():
            levels = 2 ** num_bits
            if is_weights:
                levels -= 1
            input_low = scale * level_low / level_high
            input_high = scale
        return ExportQuantize.apply(x, levels, input_low, input_high, input_low, input_high)

    return QuantizeSymmetric.apply(x, scale, level_low, level_high)
